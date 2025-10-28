import asyncio
from types import SimpleNamespace

import pytest

from backend.services import training


def test_build_face_training_options_overrides():
    options = training.build_face_training_options(
        dataset_root="custom/dataset",
        skip_download=True,
        epochs=5,
        batch=8,
        imgsz=512,
        device="cuda:0",
        project="runs/custom",
        run_name="exp42",
        base_weights="models/yolo11n.pt",
        output_weights="weights/output.pt",
    )

    assert options.dataset_root == training.settings.resolve_project_path("custom/dataset")
    assert options.skip_download is True
    assert options.epochs == 5
    assert options.batch == 8
    assert options.imgsz == 512
    assert options.device == "cuda:0"
    assert options.project == training.settings.resolve_project_path("runs/custom")
    assert options.run_name == "exp42"
    assert options.base_weights == training.settings.resolve_project_path("models/yolo11n.pt")
    assert options.output_weights == training.settings.resolve_project_path("weights/output.pt")


def test_face_training_service_prevents_parallel(monkeypatch):
    captured: list[training.FaceTrainingOptions] = []

    async def scenario() -> None:
        service = training.FaceDetectorTrainingService()
        started_event = asyncio.Event()
        release_event = asyncio.Event()

        def fake_runner(options: training.FaceTrainingOptions) -> None:
            captured.append(options)

        async def fake_to_thread(func, options):
            started_event.set()
            await release_event.wait()
            func(options)

        monkeypatch.setattr(training, "_run_face_training", fake_runner)
        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

        app = SimpleNamespace(state=SimpleNamespace(background_tasks=[]))
        options = training.build_face_training_options(epochs=1)

        first = await service.start_training(app, options)
        assert first is True
        await started_event.wait()

        second = await service.start_training(app, options)
        assert second is False

        release_event.set()
        await asyncio.sleep(0)

        assert captured == [options]
        assert service.is_running() is False
        assert app.state.background_tasks == []

    asyncio.run(scenario())
