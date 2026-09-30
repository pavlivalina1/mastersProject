import json

from src.config import CLASS_LIST_PATH, PROJECT_ROOT
from src.utils import load_json

MOBILE_LABELS_DIR = PROJECT_ROOT / "mobile" / "assets" / "labels"

# Curated display names for each raw PlantVillage class label.
# (species_en, condition_en, species_uk, condition_uk)
DISPLAY_NAMES = {
    "Apple___Apple_scab": ("Apple", "Apple scab", "Яблуня", "Парша яблуні"),
    "Apple___Black_rot": ("Apple", "Black rot", "Яблуня", "Чорна гниль"),
    "Apple___Cedar_apple_rust": ("Apple", "Cedar apple rust", "Яблуня", "Яблунево-ялівцева іржа"),
    "Apple___healthy": ("Apple", "Healthy", "Яблуня", "Здорова"),
    "Blueberry___healthy": ("Blueberry", "Healthy", "Чорниця", "Здорова"),
    "Cherry_(including_sour)___Powdery_mildew": ("Cherry", "Powdery mildew", "Вишня/Черешня", "Борошниста роса"),
    "Cherry_(including_sour)___healthy": ("Cherry", "Healthy", "Вишня/Черешня", "Здорова"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": (
        "Corn", "Cercospora / gray leaf spot", "Кукурудза", "Сіра плямистість листя (церкоспороз)"),
    "Corn_(maize)___Common_rust_": ("Corn", "Common rust", "Кукурудза", "Звичайна іржа"),
    "Corn_(maize)___Northern_Leaf_Blight": ("Corn", "Northern leaf blight", "Кукурудза", "Північний гельмінтоспоріоз"),
    "Corn_(maize)___healthy": ("Corn", "Healthy", "Кукурудза", "Здорова"),
    "Grape___Black_rot": ("Grape", "Black rot", "Виноград", "Чорна гниль"),
    "Grape___Esca_(Black_Measles)": ("Grape", "Esca (black measles)", "Виноград", "Еска (чорна кір)"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": (
        "Grape", "Leaf blight (Isariopsis)", "Виноград", "Плямистість листя (ізаріопсис)"),
    "Grape___healthy": ("Grape", "Healthy", "Виноград", "Здоровий"),
    "Orange___Haunglongbing_(Citrus_greening)": (
        "Orange", "Huanglongbing (citrus greening)", "Апельсин", "Хуанглунбін (цитрусове позеленіння)"),
    "Peach___Bacterial_spot": ("Peach", "Bacterial spot", "Персик", "Бактеріальна плямистість"),
    "Peach___healthy": ("Peach", "Healthy", "Персик", "Здоровий"),
    "Pepper,_bell___Bacterial_spot": ("Bell pepper", "Bacterial spot", "Перець солодкий", "Бактеріальна плямистість"),
    "Pepper,_bell___healthy": ("Bell pepper", "Healthy", "Перець солодкий", "Здоровий"),
    "Potato___Early_blight": ("Potato", "Early blight", "Картопля", "Рання плямистість (альтернаріоз)"),
    "Potato___Late_blight": ("Potato", "Late blight", "Картопля", "Фітофтороз"),
    "Potato___healthy": ("Potato", "Healthy", "Картопля", "Здорова"),
    "Raspberry___healthy": ("Raspberry", "Healthy", "Малина", "Здорова"),
    "Soybean___healthy": ("Soybean", "Healthy", "Соя", "Здорова"),
    "Squash___Powdery_mildew": ("Squash", "Powdery mildew", "Кабачок/Гарбуз", "Борошниста роса"),
    "Strawberry___Leaf_scorch": ("Strawberry", "Leaf scorch", "Полуниця", "Опік листя"),
    "Strawberry___healthy": ("Strawberry", "Healthy", "Полуниця", "Здорова"),
    "Tomato___Bacterial_spot": ("Tomato", "Bacterial spot", "Томат", "Бактеріальна плямистість"),
    "Tomato___Early_blight": ("Tomato", "Early blight", "Томат", "Рання плямистість (альтернаріоз)"),
    "Tomato___Late_blight": ("Tomato", "Late blight", "Томат", "Фітофтороз"),
    "Tomato___Leaf_Mold": ("Tomato", "Leaf mold", "Томат", "Бура плямистість листя (кладоспоріоз)"),
    "Tomato___Septoria_leaf_spot": ("Tomato", "Septoria leaf spot", "Томат", "Септоріоз (біла плямистість)"),
    "Tomato___Spider_mites Two-spotted_spider_mite": (
        "Tomato", "Two-spotted spider mite", "Томат", "Павутинний кліщ"),
    "Tomato___Target_Spot": ("Tomato", "Target spot", "Томат", "Кільцева плямистість"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "Tomato", "Tomato yellow leaf curl virus", "Томат", "Вірус жовтого закручування листя"),
    "Tomato___Tomato_mosaic_virus": ("Tomato", "Tomato mosaic virus", "Томат", "Вірус мозаїки томата"),
    "Tomato___healthy": ("Tomato", "Healthy", "Томат", "Здоровий"),
}

# Short, general-education care tips (Ukrainian) per raw class label — not
# prescriptive medical/pesticide dosage advice, just standard agronomic practice.
CARE_TIPS_UK = {
    "Apple___Apple_scab": "Видаліть і знищіть уражене опале листя восени; забезпечте провітрювання крони.",
    "Apple___Black_rot": "Обріжте й спаліть уражені гілки та муміфіковані плоди, уникайте пошкоджень кори.",
    "Apple___Cedar_apple_rust": "За можливості приберіть поруч ялівець (проміжного хазяїна); обробляйте навесні під час набубнявіння бруньок.",
    "Apple___healthy": "Рослина здорова — підтримуйте регулярний полив і провітрювання крони.",
    "Blueberry___healthy": "Підтримуйте кислий, добре дренований ґрунт і помірний полив.",
    "Cherry_(including_sour)___Powdery_mildew": "Забезпечте циркуляцію повітря, обрізайте загущені пагони, не перегодовуйте азотом.",
    "Cherry_(including_sour)___healthy": "Рослина здорова — підтримуйте регулярну обрізку та полив за потреби.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Дотримуйтесь сівозміни та подрібнюйте рослинні рештки після збору врожаю.",
    "Corn_(maize)___Common_rust_": "Висівайте стійкі гібриди, уникайте загущених посівів для кращого провітрювання.",
    "Corn_(maize)___Northern_Leaf_Blight": "Заорюйте рослинні рештки та дотримуйтесь сівозміни з нехазяйними культурами.",
    "Corn_(maize)___healthy": "Рослина здорова — підтримуйте сівозміну для профілактики.",
    "Grape___Black_rot": "Видаляйте муміфіковані ягоди й уражене листя, обрізайте кущ для кращої вентиляції.",
    "Grape___Esca_(Black_Measles)": "Уникайте великих обрізних ран у вологу погоду, видаляйте вражені рукави.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Покращте провітрювання листя обрізкою, видаляйте уражене листя.",
    "Grape___healthy": "Рослина здорова — підтримуйте регулярну обрізку.",
    "Orange___Haunglongbing_(Citrus_greening)": "Це карантинна хвороба — зверніться до фахівця для підтвердження; заражені дерева зазвичай видаляють, щоб зупинити поширення.",
    "Peach___Bacterial_spot": "Обирайте стійкі сорти, уникайте поливу по листю, який подовжує вологість поверхні.",
    "Peach___healthy": "Рослина здорова — підтримуйте регулярний догляд.",
    "Pepper,_bell___Bacterial_spot": "Використовуйте здорове насіння/розсаду, не працюйте з мокрим листям.",
    "Pepper,_bell___healthy": "Рослина здорова — поливайте під корінь, уникаючи листя.",
    "Potato___Early_blight": "Дотримуйтесь сівозміни, мульчуйте ґрунт, уникайте поливу по листю.",
    "Potato___Late_blight": "Швидко видаліть уражені рослини, уникайте поливу ввечері.",
    "Potato___healthy": "Рослина здорова — дотримуйтесь сівозміни.",
    "Raspberry___healthy": "Рослина здорова — вирізайте старі пагони для кращого провітрювання.",
    "Soybean___healthy": "Рослина здорова — контролюйте бур'яни та вологість ґрунту.",
    "Squash___Powdery_mildew": "Забезпечте достатню відстань між рослинами, поливайте під корінь, не по листю.",
    "Strawberry___Leaf_scorch": "Видаляйте старе уражене листя після плодоношення, уникайте загущення грядки.",
    "Strawberry___healthy": "Рослина здорова — підтримуйте мульчування та помірний полив.",
    "Tomato___Bacterial_spot": "Не працюйте з мокрим листям, дезінфікуйте інструменти між рослинами.",
    "Tomato___Early_blight": "Мульчуйте ґрунт, видаляйте нижнє уражене листя, поливайте під корінь.",
    "Tomato___Late_blight": "Швидко видаліть уражені рослини — хвороба поширюється дуже швидко у вологу погоду.",
    "Tomato___Leaf_Mold": "Знизьте вологість повітря в теплиці/під укриттям, покращте вентиляцію.",
    "Tomato___Septoria_leaf_spot": "Видаляйте нижнє уражене листя, мульчуйте, щоб зменшити бризки ґрунту на листя.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Підвищіть вологість повітря та обприскуйте листя водою — кліщі люблять сухість.",
    "Tomato___Target_Spot": "Покращте вентиляцію між рослинами, видаляйте нижнє уражене листя.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Переноситься білокрилкою — контролюйте цього шкідника, видаляйте уражені рослини.",
    "Tomato___Tomato_mosaic_virus": "Мийте руки й дезінфікуйте інструменти між рослинами — вірус легко передається контактно.",
    "Tomato___healthy": "Рослина здорова — поливайте під корінь, уникаючи листя.",
}


def build_display_labels() -> list:
    classes = load_json(CLASS_LIST_PATH)
    missing = [c for c in classes if c not in DISPLAY_NAMES]
    if missing:
        raise KeyError(f"No curated display name for classes: {missing}")
    missing_tips = [c for c in classes if c not in CARE_TIPS_UK]
    if missing_tips:
        raise KeyError(f"No care tip for classes: {missing_tips}")

    entries = []
    for idx, raw in enumerate(classes):
        species_en, condition_en, species_uk, condition_uk = DISPLAY_NAMES[raw]
        entries.append({
            "id": idx,
            "raw_label": raw,
            "species_en": species_en,
            "condition_en": condition_en,
            "species_uk": species_uk,
            "condition_uk": condition_uk,
            "healthy": condition_en.lower() == "healthy",
            "care_tip_uk": CARE_TIPS_UK[raw],
        })
    return entries


def main() -> None:
    entries = build_display_labels()
    MOBILE_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MOBILE_LABELS_DIR / "classes_display.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(entries)} class display entries to {out_path}")


if __name__ == "__main__":
    main()
