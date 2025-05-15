import logging
import cv2
import numpy as np
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, CommandHandler, ContextTypes
from io import BytesIO

# Конфигурация
TOKEN = "ЭТО МОЙ ТОКЕН"
MAX_IMAGE_SIZE = 25600  # 25MB
MAX_DIMENSION = 4500

# Настройка логгирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('bot_errors.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImageProcessor:
    @staticmethod
    def apply_cartoon_effect(img):
        """Мультяшный стиль с мягкими цветами"""
        smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        styled = cv2.stylization(smooth, sigma_s=150, sigma_r=0.25)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 2)
        edges = cv2.GaussianBlur(edges.astype(np.float32), (3, 3), 0) / 255.0
        cartoon = np.uint8(styled * (1 - edges[..., np.newaxis]) + smooth * edges[..., np.newaxis] * 0.7)
        return cartoon

    @staticmethod
    def apply_monochrome_effect(img):
        """Чёрно-белый монохром"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        film_grain = np.random.normal(0, 8, gray.shape).astype(np.uint8)
        result = cv2.addWeighted(enhanced, 0.9, film_grain, 0.1, 0)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def apply_cinematic_effect(img):
        """Кинематографический стиль"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        b = cv2.add(b, 15)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        rows, cols = img.shape[:2]
        vignette = np.ones((rows, cols), dtype=np.float32)
        x = np.linspace(-1, 1, cols)
        y = np.linspace(-1, 1, rows)
        x, y = np.meshgrid(x, y)
        vignette = 1 - (x**2 + y**2) * 0.3
        vignette = np.clip(vignette, 0.4, 1)
        vignette = cv2.GaussianBlur(vignette, (501, 501), 0)
        result = np.uint8(img * vignette[..., np.newaxis])
        grain = np.random.normal(0, 6, img.shape).astype(np.uint8)
        return cv2.addWeighted(result, 0.9, grain, 0.1, 0)

    @staticmethod
    def apply_pencil_sketch_effect(img):
        """Карандашный набросок"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        inverted_blurred = cv2.bitwise_not(blurred)
        sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def apply_vintage_effect(img):
        """Винтажный эффект"""
        kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
        sepia = cv2.transform(img, kernel)
        noise = np.random.normal(0, 15, sepia.shape).astype(np.uint8)
        noisy_img = cv2.add(sepia, noise)
        rows, cols = noisy_img.shape[:2]
        vignette = np.ones((rows, cols), dtype=np.float32)
        x = np.linspace(-1, 1, cols)
        y = np.linspace(-1, 1, rows)
        x, y = np.meshgrid(x, y)
        vignette = 1 - (x**2 + y**2) * 0.5
        vignette = np.clip(vignette, 0.3, 1)
        return np.uint8(noisy_img * vignette[..., np.newaxis])

    @staticmethod
    def process_image(image_data, style):
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Не удалось загрузить изображение")
            h, w = img.shape[:2]
            if max(h, w) > MAX_DIMENSION:
                scale = MAX_DIMENSION / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if style == 'cartoon':
                result = ImageProcessor.apply_cartoon_effect(img)
            elif style == 'monochrome':
                result = ImageProcessor.apply_monochrome_effect(img)
            elif style == 'cinematic':
                result = ImageProcessor.apply_cinematic_effect(img)
            elif style == 'pencil':
                result = ImageProcessor.apply_pencil_sketch_effect(img)
            elif style == 'vintage':
                result = ImageProcessor.apply_vintage_effect(img)
            else:
                result = img
            _, img_encoded = cv2.imencode('.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            return img_encoded.tobytes()
        except Exception as e:
            logger.error(f"Ошибка обработки: {str(e)}", exc_info=True)
            raise

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    logger.info(f"Пользователь {update.effective_user.id} запустил бота")
    help_text = (
        "🎨 <b>Бот обработки изображений</b>\n\n"
        "Доступные стили:\n"
        "/cartoon - Мультяшный стиль\n"
        "/monochrome - Чёрно-белый монохром\n"
        "/cinematic - Кинематографический стиль\n"
        "/pencil - Карандашный набросок\n"
        "/vintage - Винтажный эффект\n\n"
        "Просто отправьте фото после выбора стиля!"
    )
    await update.message.reply_text(help_text, parse_mode='HTML')

async def set_style(update: Update, context: ContextTypes.DEFAULT_TYPE, style: str):
    context.user_data['style'] = style
    style_names = {
        'cartoon': 'Мультяшный',
        'monochrome': 'Чёрно-белый монохром',
        'cinematic': 'Кинематографический',
        'pencil': 'Карандашный набросок',
        'vintage': 'Винтажный'
    }
    await update.message.reply_text(f"✅ Выбран стиль: {style_names[style]}")

async def cartoon_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_style(update, context, 'cartoon')

async def monochrome_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_style(update, context, 'monochrome')

async def cinematic_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_style(update, context, 'cinematic')

async def pencil_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_style(update, context, 'pencil')

async def vintage_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await set_style(update, context, 'vintage')

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if 'style' not in context.user_data:
            await update.message.reply_text("ℹ️ Сначала выберите стиль (/cartoon, /monochrome и т.д.)")
            return

        logger.info(f"Обработка фото от пользователя {update.effective_user.id}")
        await update.message.reply_chat_action('upload_photo')
        msg = await update.message.reply_text("🔄 Обрабатываю изображение...")
        
        photo_file = await update.message.photo[-1].get_file()
        image_data = await photo_file.download_as_bytearray()
            
        processed_image = ImageProcessor.process_image(image_data, context.user_data['style'])
        
        style_names = {
            'cartoon': "Мультяшный стиль",
            'monochrome': "Чёрно-белый монохром", 
            'cinematic': "Кинематографический стиль",
            'pencil': "Карандашный набросок",
            'vintage': "Винтажный эффект"
        }
        
        await msg.edit_text("✅ Готово!")
        await update.message.reply_photo(
            photo=BytesIO(processed_image),
            caption=f"{style_names[context.user_data['style']]}"
        )
            
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}", exc_info=True)
        await update.message.reply_text("⚠️ Произошла ошибка. Попробуйте другое изображение.")

def main():
    """Запуск бота"""
    app = Application.builder().token(TOKEN).build()
    
    # Регистрация обработчиков
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("cartoon", cartoon_style))
    app.add_handler(CommandHandler("monochrome", monochrome_style))
    app.add_handler(CommandHandler("cinematic", cinematic_style))
    app.add_handler(CommandHandler("pencil", pencil_style))
    app.add_handler(CommandHandler("vintage", vintage_style))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    logger.info("Бот успешно запущен! Ожидание сообщений...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
