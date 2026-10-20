"""Task-specific native hooks selected by the central task catalog."""
import importlib


class NativeTaskAdapter:
    def __init__(self, spec):
        self.spec = spec

    def prepare_cameras(self):
        from .cameras import install_native_camera_creation
        if self.spec.camera_environment_module:
            module = importlib.import_module('handson.sim.isaaclab.' + self.spec.camera_environment_module)
            install_native_camera_creation(module.HandsOnABCIsaacLabEnv)
    def prepare(self, output):
        if self.spec.bridge:
            module = importlib.import_module('.' + self.spec.bridge, __package__)
            module.install(dict(task=self.spec.name), output)

    def native_view(self, env, launch=True):
        if self.spec.bridge and launch:
            module = importlib.import_module('.' + self.spec.bridge, __package__)
            return module.wrap(env)
        return env

    def configure_job(self, job):
        if self.spec.force_scoring:
            from .carton import REVISION
            job['carton_force_scoring'] = dict(revision=REVISION)

    def run(self, run, env, job, output, number, seed, capture):
        if self.spec.force_scoring:
            from .carton import run_with_force
            return run_with_force(run, env, job, output, number, seed, capture)
        return run(env, job, output, number, seed, capture)

    def terminal_reason(self, returned, contact=False, env=None):
        if returned['terminated']:
            return 'native_terminated'
        if returned['truncated']:
            return 'native_truncated'
        flag = self.spec.terminal_contact_flag
        native_contact = bool(flag and env is not None and getattr(env, '_' + flag, False))
        if flag and (contact or native_contact or returned['info'].get(flag)):
            return 'native_' + flag
        return None
