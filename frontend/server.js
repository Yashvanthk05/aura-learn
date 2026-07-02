import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { createProxyMiddleware } from "http-proxy-middleware";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

const proxyOptions = {
  target: "http://localhost:7533/api",
  changeOrigin: true,
};

app.use("/api", createProxyMiddleware(proxyOptions));
app.use("/audio", createProxyMiddleware(proxyOptions));

app.use(express.static(path.join(__dirname, "dist")));

app.get("/{*path}", (req, res) => {
  res.sendFile(path.join(__dirname, "dist", "index.html"));
});

// eslint-disable-next-line no-undef
const PORT = process.env.PORT || 7532;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
