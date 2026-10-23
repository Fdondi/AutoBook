// Reads a JSON array of {tex, display} from stdin; writes a JSON array of
// null (parses) or the KaTeX error message, in the same order.
const katex = require("./node_modules/katex");

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (d) => (input += d));
process.stdin.on("end", () => {
  const items = JSON.parse(input);
  const out = items.map(({ tex, display }) => {
    try {
      katex.renderToString(tex, { displayMode: !!display, throwOnError: true, strict: "ignore" });
      return null;
    } catch (e) {
      return String(e.message || e).slice(0, 300);
    }
  });
  process.stdout.write(JSON.stringify(out));
});
