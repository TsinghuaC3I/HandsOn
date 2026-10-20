from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / "scripts" / "requirements-isaacsim-recording.txt"
INSTALLER = ROOT / "scripts" / "install_isaacsim_newton_runtime.sh"
COLLECTION_LAUNCHER = ROOT / "scripts" / "run_collection_ui.sh"


def _requirement_lines() -> list[str]:
    return [
        line.strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_isaacsim_recording_dependencies_are_exactly_pinned() -> None:
    assert _requirement_lines() == [
        "absl-py==2.4.0",
        "glfw==2.10.0",
        "zarr==2.18.3",
        "numcodecs==0.15.1",
        "av==17.1.0",
    ]
    assert all(not line.lower().startswith("numpy") for line in _requirement_lines())


def test_newton_installer_installs_and_verifies_recording_dependencies() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    assert (
        'RECORDING_REQUIREMENTS="$ROOT/scripts/requirements-isaacsim-recording.txt"'
        in source
    )
    assert 'install_recording_requirements "$PYTHON"' in source
    for distribution, version in (
        ("absl-py", "2.4.0"),
        ("glfw", "2.10.0"),
        ("zarr", "2.18.3"),
        ("numcodecs", "0.15.1"),
        ("av", "17.1.0"),
    ):
        assert source.count(f'"{distribution}": "{version}"') == 2


def test_newton_installer_verifies_without_user_site_fallbacks() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    assert source.index("export PYTHONNOUSERSITE=1") < source.index(
        "if verify_runtime; then"
    )
    assert source.count("if site.ENABLE_USER_SITE:") == 2
    assert source.count(
        'for module_name in ("absl", "glfw", "zarr", "numcodecs", "av"):'
    ) == 2
    assert source.count("if not module_path.is_relative_to(prefix):") == 2


def test_collection_launcher_requires_the_recording_dependency_marker() -> None:
    installer = INSTALLER.read_text(encoding="utf-8")
    launcher = COLLECTION_LAUNCHER.read_text(encoding="utf-8")
    marker = ".handson-newton-runtime-ready-v2"

    assert f'READY_MARKER="$TOOLCHAIN_ROOT/{marker}"' in installer
    assert f'NEWTON_READY_MARKER="$NEWTON_TOOLCHAIN_ROOT/{marker}"' in launcher
    assert (
        'LEGACY_READY_MARKER="$TOOLCHAIN_ROOT/'
        '.handson-newton-runtime-ready-v1"'
    ) in installer
    assert ".handson-newton-runtime-ready-v1" not in launcher


def test_collection_launcher_passes_one_native_isaac_runtime() -> None:
    launcher = COLLECTION_LAUNCHER.read_text(encoding="utf-8")

    assert '--isaacsim-python "$ISAACSIM_PY"' in launcher
    assert "HANDSON_ISAACSIM_V6_PYTHON" not in launcher
    assert "--isaacsim-v6-python" not in launcher
    assert "HANDSON_ENABLE_ISAACSIM_NEWTON" in launcher
    assert 'ISAACSIM_NEWTON_PY=""' in launcher


def test_collection_launcher_exposes_runtime_and_taskdev_packages() -> None:
    launcher = COLLECTION_LAUNCHER.read_text(encoding="utf-8")

    assert (
        'export PYTHONPATH="$ROOT:$ROOT/handson${clean_pythonpath:+:$clean_pythonpath}"'
        in launcher
    )


def test_collection_launcher_queues_replay_for_the_normal_night_scheduler() -> None:
    launcher = COLLECTION_LAUNCHER.read_text(encoding="utf-8")

    assert "export HANDSON_NIGHT_FORCE_RENDER_UPLOAD=0" in launcher
    assert "export HANDSON_IDLE_MAINTENANCE=1" in launcher
    assert "export HANDSON_CLOUD_RENDER_ONLY=1" in launcher
    assert "export HANDSON_REVIEW_AUTO_UPLOAD=0" in launcher
    assert "export HANDSON_BACKGROUND_V3_UPLOAD=1" in launcher


def test_legacy_runtime_is_repaired_before_v2_marker_is_published() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    repair_install = 'install_recording_requirements "$PUBLIC_ENV/bin/python"'
    assert repair_install in source
    assert source.index(repair_install) < source.index(
        '"$MAMBA" install -y -p "$PYTHON_ENV"'
    )
    assert source.count("publish_ready_marker") == 4
    assert source.index("if verify_runtime; then", source.index(repair_install)) < (
        source.index("publish_ready_marker", source.index(repair_install))
    )


def test_recording_install_forces_only_the_five_direct_pins() -> None:
    source = INSTALLER.read_text(encoding="utf-8")
    function_start = source.index("install_recording_requirements() {")
    function_end = source.index("\n}\n", function_start)
    function = source[function_start:function_end]

    forced_install = function.index('"$python" -m pip install \\')
    ignore_installed = function.index("--ignore-installed", forced_install)
    no_dependencies = function.index("--no-deps", ignore_installed)
    requirement = function.index(
        '--requirement "$RECORDING_REQUIREMENTS"', no_dependencies
    )
    dependency_resolution = function.index(
        '"$python" -m pip install --requirement "$RECORDING_REQUIREMENTS"',
        requirement,
    )
    assert (
        forced_install
        < ignore_installed
        < no_dependencies
        < requirement
        < dependency_resolution
    )
    assert "--force-reinstall" not in function
    assert source.count("install_recording_requirements \"") == 2
