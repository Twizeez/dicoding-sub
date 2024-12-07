const express = require("express");
const multer = require("multer");
const admin = require("firebase-admin");
const bodyParser = require("body-parser");
const { v4: uuidv4 } = require("uuid");
const path = require("path");
const cors = require("cors");
const tf = require("@tensorflow/tfjs-node");

// Inisialisasi Firebase Admin
const serviceAccount = require("./submissionmlgc-muhammadhafi-d28f1e964eb2.json");
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  storageBucket: "bucket-modelhafi.appspot.com",
});
const db = admin.firestore();
const bucket = admin.storage().bucket();

// Konfigurasi Multer
const upload = multer({
  limits: { fileSize: 1 * 1024 * 1024 }, // Maksimal 1MB
  fileFilter: (req, file, cb) => {
    const fileTypes = /jpeg|jpg|png/;
    const extName = fileTypes.test(path.extname(file.originalname).toLowerCase());
    if (extName) return cb(null, true);
    cb(new Error("File harus berupa gambar (jpeg/jpg/png)"));
  },
});

// Fungsi untuk memuat model dari Cloud Storage
async function loadModel() {
  const [modelFile] = await bucket.file("bucket-modelhafi/model.json").download(); 
  return await tf.loadLayersModel(tf.io.fileSystem(modelFile));
}

// Fungsi untuk melakukan prediksi
async function predictImage(buffer) {
  const image = tf.node.decodeImage(buffer, 3);
  const resizedImage = tf.image.resizeBilinear(image, [224, 224]); 
  const normalizedImage = resizedImage.div(tf.scalar(255));
  const batchedImage = normalizedImage.expandDims(0);

  const model = await loadModel();
  const prediction = model.predict(batchedImage);
  const predictionResult = prediction.dataSync(); 

  return predictionResult;
}

// Inisialisasi Express
const app = express();
app.use(cors());
app.use(bodyParser.json());

// Endpoint `/predict`
app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        status: "fail",
        message: "Gambar harus disertakan dalam permintaan",
      });
    }

    const id = uuidv4();
    const filePath = `uploads/${id}_${req.file.originalname}`;
    const file = bucket.file(filePath);

    // Upload gambar ke Cloud Storage
    await file.save(req.file.buffer, {
      metadata: { contentType: req.file.mimetype },
    });

    // Lakukan prediksi
    const predictionResult = await predictImage(req.file.buffer);
    const isCancer = predictionResult[0] > 0.5;

    const result = isCancer ? "Cancer" : "Non-cancer";
    const suggestion = isCancer
      ? "Segera periksa ke dokter!"
      : "Penyakit kanker tidak terdeteksi.";
    const createdAt = new Date().toISOString();

    // Simpan hasil prediksi ke Firestore
    await db.collection("predictions").doc(id).set({
      result,
      suggestion,
      createdAt,
      id,
    });

    res.status(200).json({
      status: "success",
      message: "Model berhasil melakukan prediksi",
      data: { id, result, suggestion, createdAt },
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({
      status: "fail",
      message: "Terjadi kesalahan dalam melakukan prediksi",
    });
  }
});

// Endpoint `/predict/histories`
app.get("/predict/histories", async (req, res) => {
  try {
    const snapshot = await db.collection("predictions").get();
    const histories = snapshot.docs.map((doc) => ({
      id: doc.id,
      history: doc.data(),
    }));

    res.status(200).json({
      status: "success",
      data: histories,
    });
  } catch (err) {
    res.status(500).json({
      status: "fail",
      message: "Gagal mengambil riwayat prediksi",
    });
  }
});

// Jalankan server
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`Server berjalan di port ${PORT}`);
});
