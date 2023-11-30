import express from 'express'
import cors from 'cors'
import Pool from "pg";
import multer from 'multer'
import tf from '@tensorflow/tfjs-node'
import sharp from 'sharp'

const PORT = 5001

const app = express()

app.use(express.urlencoded({ extended: true }));
app.use(express.json())
app.use(cors())

const pool = new Pool.Pool({
    user: "zdiroog",
    password: "password",
    host: "localhost",
    port: 5432,
    database: 'upd'
})

const colors_dict = {0: "beige", 1: "black", 2: "blue",
    3: "brown", 4: "gold", 5: "green",
    6: "grey", 7: "orange", 8: "pink",
    9: "purple", 10: "red", 11: "silver",
    12: "tan", 13: "white", 14: "yellow"}

// Настройка Multer для сохранения файлов
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// Загрузка вашей обученной модели
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const model = await tf.loadLayersModel(`file://${__dirname}/CNN/model.json`);

function determineColor(predictions) {
    // Находим ключ с максимальным значением
    const maxKey = Object.keys(predictions).reduce((a, b) => predictions[a] > predictions[b] ? a : b);

    // Получаем соответствующее значение из словаря
    const colorNumber = parseInt(maxKey);
    const colorName = colors_dict[colorNumber];

    return colorName;
}

async function preprocessImage(imageBuffer) {
    // Преобразование изображения к размеру 244x244
    const resizedImageBuffer = await sharp(imageBuffer)
        .resize(244, 244)
        .toBuffer();

    const decodedImage = tf.node.decodeImage(resizedImageBuffer, 3);
    const reshapedImage = decodedImage.reshape([244, 244, 3]);

    return reshapedImage;
}

app.get('/createImageTable', async (req, res) => {
    try {
        const client = await pool.connect();
        const query = `
      CREATE TABLE IF NOT EXISTS images (
        id SERIAL PRIMARY KEY,
        filename VARCHAR(255) NOT NULL,
        data BYTEA NOT NULL
      );
    `;

        await client.query(query);
        res.send('Таблица images успешно создана');
    } catch (err) {
        console.error('Ошибка при создании таблицы:', err);
        res.status(500).send('Внутренняя ошибка сервера');
    }
});

app.post('/uploadImage', upload.single('image'), async (req, res) => {


    try {
        const imageBuffer = req.file.buffer; // Получаем данные изображения в виде буфера

        // Предобработка изображения, например, изменение размера, нормализация, и т.д.
        const processedImage = await preprocessImage(imageBuffer);

        // Передаем изображение в модель для получения предсказания
        const batchedImage = tf.expandDims(processedImage, 0);
        const prediction = await model.predict(batchedImage);

        const predictionData = prediction.dataSync()

        // Отправляем результат предсказания обратно
        res.json({ prediction: determineColor(predictionData) });
    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});


async function startApp() {
    try {
        app.listen(PORT, () => {
            console.log('SERVER STARTED ON PORT: ', PORT)
        })
    } catch (e) {
        console.log(e)
    }
}

startApp()