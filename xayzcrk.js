'use strict';

const express = require('express');
const path = require('path');
const fs = require('fs');
const multer = require('multer');
const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require('@google/generative-ai');
const OpenAI = require('openai');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── KEY POOLS ─────────────────────────────────────────────────────────────────
// Add up to 12+ keys per provider. They rotate automatically on quota/error.
const GEMINI_KEYS = [
  process.env.GEMINI_KEY_1 || '',
  process.env.GEMINI_KEY_2 || '',
  process.env.GEMINI_KEY_3 || '',
  process.env.GEMINI_KEY_4 || '',
  process.env.GEMINI_KEY_5 || '',
  process.env.GEMINI_KEY_6 || '',
  process.env.GEMINI_KEY_7 || '',
  process.env.GEMINI_KEY_8 || '',
  process.env.GEMINI_KEY_9 || '',
  process.env.GEMINI_KEY_10 || '',
  process.env.GEMINI_KEY_11 || '',
  process.env.GEMINI_KEY_12 || '',
].filter(Boolean);

const OPENAI_KEYS = [
  process.env.OPENAI_KEY_1 || '',
  process.env.OPENAI_KEY_2 || '',
  process.env.OPENAI_KEY_3 || '',
  process.env.OPENAI_KEY_4 || '',
  process.env.OPENAI_KEY_5 || '',
  process.env.OPENAI_KEY_6 || '',
  process.env.OPENAI_KEY_7 || '',
  process.env.OPENAI_KEY_8 || '',
  process.env.OPENAI_KEY_9 || '',
  process.env.OPENAI_KEY_10 || '',
  process.env.OPENAI_KEY_11 || '',
  process.env.OPENAI_KEY_12 || '',
].filter(Boolean);

// Current key index state (in-memory rotation)
let geminiKeyIndex = 0;
let openaiKeyIndex = 0;

function getNextGeminiKey() {
  if (GEMINI_KEYS.length === 0) return null;
  const key = GEMINI_KEYS[geminiKeyIndex % GEMINI_KEYS.length];
  geminiKeyIndex = (geminiKeyIndex + 1) % GEMINI_KEYS.length;
  return key;
}

function getNextOpenAIKey() {
  if (OPENAI_KEYS.length === 0) return null;
  const key = OPENAI_KEYS[openaiKeyIndex % OPENAI_KEYS.length];
  openaiKeyIndex = (openaiKeyIndex + 1) % OPENAI_KEYS.length;
  return key;
}

// ─── SAFETY SETTINGS (Block None) ──────────────────────────────────────────────
const SAFETY_SETTINGS_BLOCK_NONE = [
  { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
];

// ─── TIMEOUT (120 minutes) ──────────────────────────────────────────────────────
const TIMEOUT_MS = 120 * 60 * 1000;

// ─── MULTER (memory storage, any file type) ─────────────────────────────────────
const storage = multer.memoryStorage();
const upload = multer({
  storage,
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB per file
});

// ─── MIDDLEWARE ─────────────────────────────────────────────────────────────────
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'auto-ai')));

// ─── REDIRECT ROOT TO /auto-ai ──────────────────────────────────────────────────
app.get('/auto-ai', (req, res) => {
  res.redirect('/ipa/auto-ai');
});

// ─── MAIN PAGE ──────────────────────────────────────────────────────────────────
app.get('/ipa/auto-ai', (req, res) => {
  res.sendFile(path.join(__dirname, 'auto-ai', 'ipa-auto-ai.html'));
});

// ─── HELPER: isQuotaError ───────────────────────────────────────────────────────
function isQuotaError(err) {
  if (!err) return false;
  const msg = (err.message || '').toLowerCase();
  const code = err.status || err.statusCode || (err.error && err.error.code) || 0;
  return (
    code === 429 ||
    code === 402 ||
    msg.includes('quota') ||
    msg.includes('rate limit') ||
    msg.includes('exceeded') ||
    msg.includes('billing') ||
    msg.includes('insufficient_quota') ||
    msg.includes('resource_exhausted') ||
    msg.includes('too many requests') ||
    msg.includes('limit reached')
  );
}

// ─── GEMINI CALL with key rotation ──────────────────────────────────────────────
async function callGemini(modelId, prompt, fileParts) {
  let attempts = 0;
  const maxAttempts = Math.max(GEMINI_KEYS.length, 1);

  while (attempts < maxAttempts) {
    const apiKey = getNextGeminiKey();
    if (!apiKey) throw new Error('No Gemini API keys configured. Please set GEMINI_KEY_1 in environment variables.');

    try {
      const genAI = new GoogleGenerativeAI(apiKey);
      const model = genAI.getGenerativeModel({
        model: modelId,
        safetySettings: SAFETY_SETTINGS_BLOCK_NONE,
      });

      const parts = [];
      if (fileParts && fileParts.length > 0) {
        parts.push(...fileParts);
      }
      parts.push({ text: prompt });

      const result = await model.generateContent({ contents: [{ role: 'user', parts }] });
      const response = result.response;
      return response.text();
    } catch (err) {
      attempts++;
      if (isQuotaError(err) && attempts < maxAttempts) {
        console.warn(`Gemini key ${attempts} exhausted/limited, trying next key...`);
        continue;
      }
      throw err;
    }
  }
  throw new Error('All Gemini API keys exhausted or rate-limited.');
}

// ─── OPENAI CALL with key rotation ──────────────────────────────────────────────
async function callOpenAI(modelId, prompt, fileDataList) {
  let attempts = 0;
  const maxAttempts = Math.max(OPENAI_KEYS.length, 1);

  while (attempts < maxAttempts) {
    const apiKey = getNextOpenAIKey();
    if (!apiKey) throw new Error('No OpenAI API keys configured. Please set OPENAI_KEY_1 in environment variables.');

    try {
      const openai = new OpenAI({ apiKey, timeout: TIMEOUT_MS });

      const content = [];

      // Add files as inline data if any
      if (fileDataList && fileDataList.length > 0) {
        for (const fd of fileDataList) {
          if (fd.mimeType && fd.mimeType.startsWith('image/')) {
            content.push({
              type: 'image_url',
              image_url: { url: `data:${fd.mimeType};base64,${fd.data}` },
            });
          } else {
            // Non-image: include as text description
            content.push({
              type: 'text',
              text: `[Attached file: ${fd.name} (${fd.mimeType}, ${Math.round(fd.size / 1024)} KB)]`,
            });
          }
        }
      }

      content.push({ type: 'text', text: prompt });

      const completion = await openai.chat.completions.create({
        model: modelId,
        messages: [{ role: 'user', content }],
        max_tokens: 4096,
      });

      return completion.choices[0].message.content;
    } catch (err) {
      attempts++;
      if (isQuotaError(err) && attempts < maxAttempts) {
        console.warn(`OpenAI key ${attempts} exhausted/limited, trying next key...`);
        continue;
      }
      throw err;
    }
  }
  throw new Error('All OpenAI API keys exhausted or rate-limited.');
}

// ─── CHAT ENDPOINT ──────────────────────────────────────────────────────────────
app.post('/api/chat', upload.array('files', 20), async (req, res) => {
  // Set response timeout
  req.setTimeout(TIMEOUT_MS);
  res.setTimeout(TIMEOUT_MS);

  try {
    const { message, model, questionContext } = req.body;
    const files = req.files || [];

    if (!message && files.length === 0) {
      return res.status(400).json({ error: 'Pesan tidak boleh kosong.' });
    }

    const selectedModel = model || 'gemini-3.5-flash';
    const prompt = questionContext
      ? `Konteks soal IPA:\n${questionContext}\n\nPertanyaan/Permintaan:\n${message || 'Tolong bantu analisis soal ini.'}`
      : message || 'Tolong analisis file yang dilampirkan.';

    let responseText = '';

    if (selectedModel.startsWith('gemini')) {
      // Build Gemini file parts
      const fileParts = files.map((file) => ({
        inlineData: {
          mimeType: file.mimetype,
          data: file.buffer.toString('base64'),
        },
      }));
      responseText = await callGemini(selectedModel, prompt, fileParts);
    } else {
      // OpenAI
      const fileDataList = files.map((file) => ({
        name: file.originalname,
        mimeType: file.mimetype,
        data: file.buffer.toString('base64'),
        size: file.size,
      }));
      responseText = await callOpenAI(selectedModel, prompt, fileDataList);
    }

    res.json({ response: responseText });
  } catch (err) {
    console.error('Chat error:', err);
    const msg = err.message || 'Terjadi kesalahan pada server.';
    res.status(500).json({ error: msg });
  }
});

// ─── HEALTH CHECK ───────────────────────────────────────────────────────────────
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    geminiKeys: GEMINI_KEYS.length,
    openaiKeys: OPENAI_KEYS.length,
    timestamp: new Date().toISOString(),
  });
});

// ─── START SERVER ────────────────────────────────────────────────────────────────
if (require.main === module) {
  const server = app.listen(PORT, () => {
    console.log(`✅ xayzcrk AI server running at http://localhost:${PORT}`);
    console.log(`   Gemini keys loaded: ${GEMINI_KEYS.length}`);
    console.log(`   OpenAI keys loaded: ${OPENAI_KEYS.length}`);
    console.log(`   Redirect: / → /auto-ai`);
  });
  server.setTimeout(TIMEOUT_MS);
}

module.exports = app;
