// Replace pandoc's \( \) and \[ \] math spans with MathJax SVG so Chrome prints real LaTeX without network access.
const fs = require("fs");
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
require("mathjax-full/js/input/tex/base/BaseConfiguration.js");
require("mathjax-full/js/input/tex/ams/AmsConfiguration.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");

const [inp, out] = process.argv.slice(2);
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const doc = mathjax.document(fs.readFileSync(inp, "utf8"), {
  InputJax: new TeX({ packages: ["base", "ams"], inlineMath: [["\\(", "\\)"]], displayMath: [["\\[", "\\]"]] }),
  OutputJax: new SVG({ fontCache: "global" }),
});
doc.render();
fs.writeFileSync(out, adaptor.doctype(doc.document) + adaptor.outerHTML(adaptor.root(doc.document)));
