let modelo;
const vocabSize = 1000;
const maxLen = 15;

function simpleTokenizer(frase) {
  const palabras = frase.toLowerCase().replace(/[^\w\s]/g, '').split(" ");
  const indices = palabras.map(p => (p.charCodeAt(0) || 0) % vocabSize);
  return indices;
}

function procesarEntrada(texto) {
  let secuencia = simpleTokenizer(texto);
  while (secuencia.length < maxLen) secuencia.push(0);
  return tf.tensor2d([secuencia.slice(0, maxLen)]);
}

async function cargarModelo() {
  modelo = await tf.loadLayersModel("modelo_emociones_tfjs/model.json");
  console.log("✅ Modelo cargado");
}
cargarModelo();

async function predecir() {
  const texto = document.getElementById("entrada").value;
  const entrada = procesarEntrada(texto);
  const resultado = await modelo.predict(entrada).data();
  const emocion = resultado[0] > 0.5 ? "POSITIVO 😊" : "NEGATIVO 😞";
  document.getElementById("resultado").textContent = `Resultado: ${emocion}`;
}
