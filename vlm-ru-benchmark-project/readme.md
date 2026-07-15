Практический проект VK Education: сравнение и дообучение визуально-языковых моделей (VLM) на бенчмарках GQA-ru и MMBench-ru от deepvk

## Название проекта
Сравнительный анализ и повышение производительности визуально-языковых моделей на бенчмарках GQA-ru и MMBench-ru

## Цель проекта
Получить наиболее высокую метрику качества на открытых русскоязычных бенчмарках GQA-ru и MMBench-ru за счёт сравнения существующих визуально-языковых моделей (baseline-решений deepvk) с более новыми мультиязычными VLM и, при наличии времени, дообучения модели на открытых данных VK.

## Задачи
1. Изучить открытые датасеты и baseline-модели из коллекции deepvk Vision-Language Modeling (`GQA-ru`, `MMBench-ru`, `LLaVA-Instruct-ru`, `llava-saiga-8b`, `llava-gemma-2b-lora`).
2. Развернуть пайплайн оценки моделей на базе библиотеки `lmms-eval`.
3. Получить baseline-метрики существующих моделей deepvk на GQA-ru и MMBench-ru.
4. Оценить zero-shot производительность современной компактной мультиязычной VLM (Qwen2.5-VL-3B-Instruct) на тех же бенчмарках и сравнить с baseline.
5. При наличии ресурсов — дообучить (LoRA) модель на подвыборке `LLaVA-Instruct-ru` и повторно оценить.
6. Задокументировать методологию, использованные данные и результаты сравнения моделей.

## Ожидаемые результаты
- Таблица сравнения метрик (ExactMatch/GPTEvalScore) для всех протестированных моделей на GQA-ru и MMBench-ru.
- Итоговая модель (или конфигурация модели + промпт-стратегия), показавшая наилучший результат.
- Воспроизводимый код пайплайна оценки (и дообучения, если было) в репозитории на GitHub.
- Файл с подробным описанием решения: постановка задачи, использованные данные, ход эксперимента, метрики, выводы.

## Способы и средства достижения результата

**Среда вычислений:** Google Colab (бесплатный T4 GPU, 16GB VRAM)

**Стек инструментов:**

| Задача | Инструмент |
|---|---|
| Оценка моделей на бенчмарках | [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval) |
| Загрузка датасетов/моделей | `datasets`, `transformers` (HuggingFace) |
| Дообучение | `peft` (LoRA), `transformers.Trainer` |
| Версионирование и хранение кода | Git + GitHub |
| Трекинг экспериментов | таблица метрик в README / CSV |

**Модели для сравнения:**
1. `deepvk/llava-gemma-2b-lora` (3B) — baseline
2. `deepvk/llava-saiga-8b` (8B) — baseline
3. `Qwen/Qwen2.5-VL-3B-Instruct` — современная мультиязычная VLM для сравнения

## Изучение открытой базы датасетов

### GQA-ru
Переведённая на русский версия бенчмарка GQA (визуальные вопросы-ответы, формат similar to `lmms-lab/GQA`).

- **Train split:** 27 519 изображений, 40 000 вопросов
- **Test split (testdev):** 398 изображений, 12 216 вопросов
- **Ключевые поля:** `question`, `answer`, `fullAnswer`, `imageId`, `isBalanced`, `groups`, `types` (structural/semantic/detailed), `annotations`
- **Промпт при инференсе:** `"Ответь одним словом."` (пост-промпт, который использовали авторы deepvk при обучении своих моделей)
- **Метрика:** ExactMatch — сгенерированное слово должно точно совпасть с эталонным ответом

### MMBench-ru
Переведённая версия MMBench (формат `lmms-lab/MMBench_EN`), multiple-choice вопросы по изображениям.

- **Размер:** 3 910 примеров
- **Формат:** вопрос + варианты ответов (single-choice, 2–4 опции), модель должна выбрать букву
- **Метрика:** GPTEvalScore — если ответ модели совпадает с буквой напрямую, засчитывается как ExactMatch; если ответ развёрнутый — сверяется через OpenAI API. Без API-ключа метрика сводится к классическому ExactMatch

### Команды для запуска оценки (lmms-eval)

```bash
# GQA-ru
accelerate launch -m lmms_eval --model llava_hf \
  --model_args pretrained="deepvk/llava-saiga-8b" \
  --tasks gqa-ru --batch_size 1 \
  --log_samples --log_samples_suffix llava-saiga-8b --output_path ./logs/

# MMBench-ru
accelerate launch -m lmms_eval --model llava_hf \
  --model_args pretrained="deepvk/llava-saiga-8b" \
  --tasks mmbench_ru_dev --batch_size 1 \
  --log_samples --log_samples_suffix llava-saiga-8b --output_path ./logs/
```

### Вывод
Оба бенчмарка небольшие по объёму (особенно test-сплиты), что реалистично прогнать на бесплатном Colab T4 без квантизации проблем — даже 8B-модель влезет для инференса. Для дообучения (шаг 6) стоит использовать train-сплит GQA-ru (40k вопросов) и `LLaVA-Instruct-ru`.