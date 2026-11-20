"""Similarity alignment and bounded ID framing; no learned codec or demographics."""
import numpy as np
from PIL import Image
from scipy import ndimage
from .foreground import BACKGROUND,alpha_bytes,resize_float

FRAMES={"conservative":{"interocular":22.,"eye_y":34.,"margin":4.},
        "balanced":{"interocular":25.,"eye_y":35.,"margin":2.},
        "tight":{"interocular":28.,"eye_y":36.,"margin":1.}}


def eye_coordinates(points):
    p=np.asarray(points,dtype=float)
    if p.shape!=(5,2) or not np.isfinite(p).all():raise ValueError("Invalid five-point landmarks")
    eye=.5*(p[0]+p[1]);vector=p[1]-p[0];distance=float(np.linalg.norm(vector))
    if distance<=1 or vector[0]<=0:raise ValueError("Invalid interocular geometry")
    u=vector/distance;rotation=np.array([[u[0],u[1]],[-u[1],u[0]]])
    return eye,distance,rotation


def apply_transform(points,matrix):
    p=np.asarray(points,dtype=float)
    return p@np.asarray(matrix)[:,:2].T+np.asarray(matrix)[:,2]


def head_envelope(points,alpha,dense=None):
    """Provisional head bounds from face mesh plus person alpha, not a hair detector."""
    eye,d,r=eye_coordinates(points);p=np.asarray(points)
    if dense is not None:
        mesh=np.asarray(dense,dtype=float)
        if mesh.ndim!=2 or mesh.shape[0]<455 or mesh.shape[1]!=2 or not np.isfinite(mesh).all():raise ValueError("Invalid dense landmarks")
        chin=mesh[152];face=np.array([mesh[i] for i in (10,152,234,454)])
    else:
        chin=.5*(p[3]+p[4])+r[1]*.65*d
        face=np.stack([eye-r[1]*.85*d,chin,eye-r[0]*.95*d,eye+r[0]*.95*d])
    a=np.asarray(alpha,dtype=float)
    if a.ndim!=2 or not np.isfinite(a).all() or a.min()<0 or a.max()>1:raise ValueError("Invalid native alpha")
    yy,xx=np.indices(a.shape);xy=np.stack([xx,yy],axis=-1);local=(xy-eye)@r.T
    mouth_local=(.5*(p[3]+p[4])-eye)@r.T
    # Head search excludes the lower shoulders. Raised hands may still enter
    # the envelope: those are explicitly reviewed by independent content checks.
    roi=local[...,1]<=mouth_local[1]+.12*d
    labels,_=ndimage.label(a>=.15,structure=np.ones((3,3)))
    nose=np.rint(p[2]).astype(int);nx,ny=np.clip(nose,[0,0],[a.shape[1]-1,a.shape[0]-1])
    component=labels[ny,nx];selected=roi&(labels==component)&(a>=.15) if component else np.zeros_like(roi)
    coords=xy[selected]
    if len(coords):
        # Include matte wisps at their native pixel locations; preserve actual
        # alpha values for rendering (the .15 threshold affects bounds only).
        combined=np.concatenate([coords,face])
    else:combined=face
    transformed=(combined-eye)@r.T
    bounds=[float(transformed[:,0].min()),float(transformed[:,1].min()),float(transformed[:,0].max()),float(transformed[:,1].max())]
    chin_local=(chin-eye)@r.T;bounds[3]=max(bounds[3],float(chin_local[1])+.08*d)
    boundary=bool(selected[0].any() or selected[:,0].any() or selected[:,-1].any())
    return {"bounds_eye_coordinates":bounds,"chin_source":chin.tolist(),"source_boundary_contact":boundary,
        "alpha_component_found":bool(component),"dense_head_available":dense is not None,
        "source_head_pixels":int(selected.sum()),"interocular_distance":d}


def crop_transform(points,envelope,frame):
    if frame not in FRAMES:raise ValueError("Unknown frame")
    eye,d,rotation=eye_coordinates(points);spec=FRAMES[frame];x0,y0,x1,y1=envelope["bounds_eye_coordinates"]
    cx=31.5;ey=spec["eye_y"];margin=spec["margin"]
    limits=[spec["interocular"]/d]
    for available,extent in ((cx-margin,-x0),(63-margin-cx,x1),(ey-margin,-y0),(79-margin-ey,y1)):
        if extent>0:limits.append(available/extent)
    scale=float(min(limits))
    if not np.isfinite(scale) or scale<=0:raise ValueError("No feasible similarity crop")
    matrix=np.column_stack([scale*rotation,np.array([cx,ey])-scale*rotation@eye])
    return {"frame":frame,"matrix_source_to_output":matrix.tolist(),"scale":scale,
        "rotation_degrees":float(-np.degrees(np.arctan2(rotation[0,1],rotation[0,0]))),
        "target_eye_midpoint":[cx,ey],"actual_interocular_pixels":scale*d,
        "requested_interocular_pixels":spec["interocular"],"head_fit_limited":scale*d<spec["interocular"]-1e-8,
        "chin_output":apply_transform([envelope["chin_source"]],matrix)[0].tolist(),
        "output_size":[64,80],"nonuniform_scaling":False}


def render_crop(image,alpha,transform,supersample=4):
    import cv2
    a=np.asarray(alpha,dtype=np.float32)
    if a.shape!=(image.height,image.width) or not np.isfinite(a).all() or a.min()<0 or a.max()>1:raise ValueError("Alpha/source mismatch")
    gray=np.asarray(image.convert("L"),dtype=np.float32)/255
    composite=a*gray+(1-a)*BACKGROUND
    matrix=np.asarray(transform["matrix_source_to_output"],dtype=np.float64)
    linear=matrix[:,:2]
    if not np.allclose(linear.T@linear,np.eye(2)*np.linalg.det(linear),rtol=1e-7,atol=1e-9):raise ValueError("Crop must be a uniform similarity transform")
    high=matrix.copy();high[:,:2]*=supersample;high[:,2]=high[:,2]*supersample+(supersample-1)/2
    size=(64*supersample,80*supersample)
    # Composite once before filtering, with a gray border. Alpha is transformed
    # independently for metadata only, never multiplied into the target again.
    target_hi=cv2.warpAffine(composite,high,size,flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=BACKGROUND)
    alpha_hi=cv2.warpAffine(a,high,size,flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    support=cv2.warpAffine(np.ones_like(a),high,size,flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    target=np.clip(resize_float(target_hi),0,1);out_alpha=np.clip(resize_float(alpha_hi),0,1)
    return Image.fromarray(alpha_bytes(target)),Image.fromarray(alpha_bytes(out_alpha)),{
        "padding_fraction":float(1-support.mean()),"foreground_area":float(out_alpha.mean()),
        "interpolation":"OpenCV cubic to 4x pixel-center-aligned canvas, Pillow F LANCZOS to 64x80",
        "composite":"native uint8-alpha portrait on 128/255, before filtering; gray outside source"}
