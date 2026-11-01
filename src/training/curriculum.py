"""Version dispatch. Frozen v2 scheduling remains available unchanged."""
from importlib import import_module


def implementation(config):
    version = config.get("version")
    if type(version) is not int or version not in (2, 3):
        raise ValueError("only curriculum versions 2 and 3 are supported")
    return import_module(f"src.training.curriculum_v{version}")


def descriptor_scene(descriptor, role, attempt):
    if "placement_search" in descriptor:
        from src.training.scene_placement import candidate_scene
        return candidate_scene(descriptor, attempt)
    options = dict(strata=descriptor.get("strata"))
    if descriptor["version"] == 3:
        options["release_revision"] = descriptor.get("release_revision",1)
    return implementation(descriptor).scene(descriptor["course"], descriptor["seed"], role,
        descriptor["case"], attempt, **options)
