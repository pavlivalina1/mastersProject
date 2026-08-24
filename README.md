# Інтелектуальна система класифікації захворювань сільськогосподарських культур із використанням методів пояснюваного штучного інтелекту

Дипломний проєкт: порівняння кількох CNN-архітектур для класифікації хвороб рослин за
фото листя, перевірка узагальнювальної здатності моделей та пояснення їхніх рішень
методами explainable AI (XAI).

## Дані

- **PlantVillage** (`spMohanty/PlantVillage-Dataset`, папка `raw/color`) - основний датасет
  для навчання/валідації/тесту. ~38 класів культур/хвороб.
- **PlantDoc** (`pratikkayal/PlantDoc-Dataset`) - реальні польові фото, використовуються
  лише для out-of-distribution тесту узагальнювальної здатності (класи змаплено на
  найближчі класи PlantVillage, див. `PLANTDOC_TO_PLANTVILLAGE` у
  `src/data/prepare_dataset.py`).

Обидва датасети завантажуються як git-репозиторії у `data/raw/`. Через те, що частина
файлів у цих репозиторіях містить символи, заборонені у Windows (напр. `?` у назвах,
запозичених з URL), стандартний `git checkout` на Windows завершується помилкою - тому
файли видобуваються напряму з git-дерева скриптом, який санітизує неприпустимі символи
в іменах (`git ls-tree` + `git cat-file`), а не через звичайний checkout.

## Встановлення

```bash
pip install -r requirements.txt
```

PyTorch/torchvision встановлені у CPU-версії (GPU відсутній). Тому всі
5 моделей використовують transfer learning (заморожений backbone + fine-tune останніх
шарів) на зменшеній підвибірці датасету (`IMAGES_PER_CLASS` у `src/config.py`), щоб
навчання реально завершувалось за прийнятний час на CPU.

## Моделі

1. `custom_cnn` - власна невелика CNN, навчена з нуля (базова лінія без transfer learning)
2. `resnet18` - ImageNet pretrained, fine-tune `layer4` + голови
3. `mobilenet_v2` - ImageNet pretrained, легка мобільна архітектура
4. `efficientnet_b0` - ImageNet pretrained
5. `shufflenet_v2` - ImageNet pretrained, найлегша модель (порівняння швидкість/точність)

## Як відтворити експерименти

```bash
# Швидка перевірка коду (1 епоха, 20 зобр./клас) перед повним запуском
python run_all_experiments.py --smoke

# Повний запуск: підготовка даних -> навчання 5 моделей -> оцінка ->
# узагальнювальна здатність (CV, OOD, robustness) -> XAI -> звіт
python run_all_experiments.py
```

Або крок за кроком:

```bash
python -m src.data.prepare_dataset          # побудова train/val/test/ood маніфестів
python -m src.train --model resnet18        # навчання однієї моделі
python -m src.evaluate --model resnet18     # оцінка на тестовій вибірці
python -m src.generalization                # CV + OOD + robustness для всіх моделей
python -m src.run_xai                       # Grad-CAM/Grad-CAM++/LIME/SHAP + faithfulness
python -m src.generate_report               # зведений reports/comparison_report.md
```

## Структура результатів

- `experiments/results/{model}_metrics.json` - accuracy/precision/recall/F1, розмір моделі,
  латентність інференсу
- `experiments/results/{model}_confusion.png`, `{model}_curves.png`
- `experiments/results/cv_results.json` - k-fold крос-валідація
- `experiments/results/ood_results.json` - точність на PlantDoc (out-of-distribution)
- `experiments/results/robustness_results.json` - точність під спотвореннями (шум,
  розмиття, яскравість, поворот)
- `experiments/results/xai/` - Grad-CAM/Grad-CAM++/LIME візуалізації + `faithfulness_scores.json`
  (insertion/deletion AUC для кожного методу XAI)

## Перевірка узагальнювальної здатності

Реалізовано трьома незалежними способами (`src/generalization.py`):

1. **K-fold крос-валідація** - оцінка стабільності метрик на різних розбиттях даних.
2. **OOD-тест на PlantDoc** - та ж модель, навчена на лабораторних фото PlantVillage,
   перевіряється на реальних польових фото іншого розподілу.
3. **Robustness-тест** - контрольовані спотворення (шум, розмиття, яскравість, поворот)
   різної інтенсивності, вимірюється деградація точності.

## XAI

`src/xai/` реалізує Grad-CAM, Grad-CAM++, LIME та SHAP, а також кількісну оцінку
"вірності" пояснень через insertion/deletion AUC (`src/xai/faithfulness.py`) - це дає
змогу не лише показати теплокарти, а й порівняти методи пояснення між собою кількісно.
