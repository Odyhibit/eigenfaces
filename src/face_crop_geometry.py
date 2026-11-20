"""Phase 10B geometry and feature validation, independent of person/hair masks."""
import numpy as np

from .id_portrait_geometry import eye_coordinates, apply_transform

VARIANTS = {"Face20": 20., "Face24": 24., "Face28": 28.}
# MediaPipe Face Mesh topology. Iris points are deliberately unnecessary.
FEATURES = {
    "eyebrows": [70,63,105,66,107,55,65,52,53,46,336,296,334,293,300,276,283,282,295,285],
    "eyes": [33,160,158,133,153,144,362,385,387,263,373,380],
    "nose": [168,6,197,195,5,4,1,2,98,327,129,358],
    "mouth": [61,40,37,0,267,270,291,321,314,17,84,91,78,13,308,14],
    "chin": [148,152,377,176,400],
    "contours": [234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454],
    "forehead": [10,109,67,103,54,338,297,332,284],
}
CRITICAL = ("eyebrows", "eyes", "nose", "mouth")


def feature_points(points, dense=None):
    p = np.asarray(points, float)
    eye, d, r = eye_coordinates(p)
    if dense is not None:
        mesh = np.asarray(dense, float)
        if mesh.ndim != 2 or mesh.shape[0] < 468 or mesh.shape[1] != 2 or not np.isfinite(mesh).all():
            raise ValueError("Invalid dense landmarks")
        result = {k: mesh[v] for k,v in FEATURES.items()}
        # Include original eyes and mouth corners so dense hallucinations cannot
        # silently move these known critical landmarks out of the validation.
        result["eyes"] = np.vstack([result["eyes"], p[:2]])
        result["mouth"] = np.vstack([result["mouth"], p[3:]])
        result["nose"] = np.vstack([result["nose"], p[2]])
        return result, "dense_plus_original_five"
    def local(values):
        return eye + np.asarray(values)*d @ r
    # Fallback estimates are explicitly uncertain, never certified feature fits.
    mouth = ((p[3:5].mean(0)-eye) @ r.T)/d
    result = {
        "eyes": local([[-.65,-.12],[.65,.12]]),
        "eyebrows": local([[-.8,-.40],[.8,-.15]]),
        "nose": np.vstack([p[2],local([[-.25,.15],[.25,.8]])]),
        "mouth": np.vstack([p[3:],local([mouth+[-.55,-.18],mouth+[.55,.22]])]),
        "chin": local([mouth+[-.20,.65],mouth+[.20,.65]]),
        "contours": local([[-1.,0],[-.65,1.5],[.65,1.5],[1.,0]]),
        "forehead": local([[-.6,-.65],[.6,-.65]]),
    }
    return result, "five_point_estimated_extents"


def similarity(points, requested, eye_y):
    if requested not in VARIANTS.values() or not np.isfinite(eye_y):
        raise ValueError("Unsupported crop specification")
    eye, d, rotation = eye_coordinates(points)
    scale = requested/d
    matrix = np.column_stack([scale*rotation, [31.5,eye_y]-scale*rotation@eye])
    actual = float(np.linalg.norm(np.diff(apply_transform(np.asarray(points)[:2],matrix),axis=0)))
    return {"matrix_source_to_output": matrix.tolist(), "scale": scale,
            "requested_interocular_pixels": requested, "actual_interocular_pixels": actual,
            "target_eye_midpoint": [31.5,float(eye_y)], "nonuniform_scaling": False,
            "head_fit_limited": False, "output_size": [64,80]}


def validate_features(features, transform, source_size):
    matrix = transform["matrix_source_to_output"]
    result = {}
    for name, source in features.items():
        source = np.asarray(source)
        output = apply_transform(source,matrix)
        margins = np.column_stack([output[:,0]+.5,63.5-output[:,0],output[:,1]+.5,79.5-output[:,1]])
        inside = (margins >= -1e-8).all(1)
        source_inside = ((source >= -.5) & (source <= np.array(source_size)-.5)).all(1)
        result[name] = {"points_output":output.tolist(), "bounds_output":[*output.min(0),*output.max(0)],
                        "minimum_boundary_margin_pixels":float(margins.min()),
                        "clipped":bool(not inside.all()), "inside_fraction":float(inside.mean()),
                        "source_clipped":bool(not source_inside.all())}
        result[name]["bounds_output"] = [float(x) for x in result[name]["bounds_output"]]
    critical = any(result[k]["clipped"] or result[k]["source_clipped"] for k in CRITICAL)
    return {"features":result, "critical_feature_failure":critical,
            "chin_fits":not (result["chin"]["clipped"] or result["chin"]["source_clipped"]),
            "contours_usable":result["contours"]["inside_fraction"] >= .85,
            "crop_feasibility":"failed" if critical or result["chin"]["clipped"] else "feasible"}


def choose_eye_lines(items):
    """One eye Y per variant, from geometry alone, frozen before assessment.

    Maximize complete eyebrow-through-chin vertical fits, then critical fits;
    ties prefer 31. Horizontal clipping cannot be fixed by eye Y or scale caps.
    """
    result = {}; diagnostics = {}
    for name, requested in VARIANTS.items():
        trials = []
        for y in range(22,35):
            complete = critical = 0
            for points,features in items:
                matrix = similarity(points,requested,y)["matrix_source_to_output"]
                fit = {}
                for key in (*CRITICAL,"chin"):
                    q = apply_transform(features[key],matrix)
                    fit[key] = bool((q[:,1]>=.5).all() and (q[:,1]<=78.5).all())
                critical += all(fit[k] for k in CRITICAL)
                complete += all(fit.values())
            trials.append({"eye_y":y,"vertical_complete":complete,"vertical_critical":critical})
        best = max(trials,key=lambda t:(t["vertical_complete"],t["vertical_critical"],-abs(t["eye_y"]-31)))
        result[name] = float(best["eye_y"]); diagnostics[name] = {"selected":best,"trials":trials}
    return result,diagnostics
