"""Утилита для добавления детализированного логирования в AI-детектор."""
import logging
from pathlib import Path
from typing import Optional


def patch_ai_detector() -> bool:
    """Добавляет детальное логирование в AI-детектор."""

    candidates = [
        Path("/app/backend/services/ai_detector.py"),
        Path(__file__).resolve().parent / "backend/services/ai_detector.py",
    ]

    ai_detector_path: Optional[Path] = None
    for candidate in candidates:
        if candidate.exists():
            ai_detector_path = candidate
            break

    if ai_detector_path is None:
        print("❌ Файл ai_detector.py не найден")
        return False

    content = ai_detector_path.read_text(encoding="utf-8")

    # Проверяем, не патчили ли уже
    if "DEBUG_PATCH_ADDED" in content:
        print("✅ Патч уже установлен")
        return True

    # Добавляем логирование в process_frame
    patch_code = """
    # === DEBUG_PATCH_ADDED ===
    logger.debug("[%s] DEBUG: Начало process_frame, размер кадра: %s", self.camera_name, frame.shape if frame is not None else "None")
    """

    # Находим место для вставки - после объявления переменных в process_frame
    target_line = "def process_frame("
    insert_pos = content.find(target_line)
    if insert_pos == -1:
        print("❌ Не найдена функция process_frame")
        return False

    signature_end = content.find(") ->", insert_pos)
    if signature_end == -1:
        print("❌ Не удалось определить конец сигнатуры process_frame")
        return False
    func_start = content.find("\n", signature_end) + 1

    # Вставляем наш код
    patched_content = content[:func_start] + "    " + patch_code + "\n    " + content[func_start:]

    # Добавляем логирование после детекции
    det_log_patch = """
    # DEBUG: Логирование результатов детектора
    if det_res is not None:
        boxes = getattr(det_res, "boxes", None)
        if boxes is not None:
            logger.debug("[%s] DEBUG: Детектор вернул %d боксов", self.camera_name, len(boxes))
            for i, box in enumerate(boxes):
                try:
                    cls_idx = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.det_names.get(cls_idx, "unknown")
                    logger.debug("[%s] DEBUG: Бокс %d: %s (conf: %.3f)", self.camera_name, i, label, conf)
                except Exception as e:  # pragma: no cover - только диагностика
                    logger.debug("[%s] DEBUG: Ошибка парсинга бокса %d: %s", self.camera_name, i, e)
        else:
            logger.debug("[%s] DEBUG: Детектор не вернул боксов", self.camera_name)
    else:
        logger.debug("[%s] DEBUG: Детектор вернул None", self.camera_name)
    """

    # Вставляем после получения det_res
    det_res_pos = patched_content.find("det_res = self.det(frame, **det_kwargs)[0]")
    if det_res_pos != -1:
        line_end = patched_content.find("\n", det_res_pos)
        patched_content = patched_content[:line_end] + det_log_patch + patched_content[line_end:]

    # Добавляем логирование в конце
    end_log_patch = """
    # DEBUG: Финальные результаты
    logger.info("[%s] DEBUG: ИТОГО - люди: %d, машины: %d, телефон: %s (conf: %.3f)", 
                self.camera_name, len(people_data), len(car_events), phone_usage, best_conf)
    """

    return_pos = patched_content.find("return (")
    if return_pos != -1:
        patched_content = patched_content[:return_pos] + end_log_patch + "\n    " + patched_content[return_pos:]

    # Сохраняем изменения
    ai_detector_path.write_text(patched_content, encoding="utf-8")
    print("✅ Патч успешно установлен")
    return True


if __name__ == "__main__":
    patch_ai_detector()
