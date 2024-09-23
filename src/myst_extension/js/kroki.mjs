import pako from 'pako';
import fs from 'fs';
import path from 'path';

function compress_b64(raw_text) {
  const data = new TextEncoder().encode(raw_text);
  const compressed = pako.deflate(data, { level: 9 });
  const result = Buffer.from(compressed)
    .toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_');
  return result
}

const krokiDirective = {
    name: 'kroki',
    doc: 'Call krobi to creates diagrams from textual descriptions!',
    arg: {
      type: String,
    },
    body: {
      type: String,
    },
    options: {
      src: { type: String, doc: 'language type of input image, e.g. graphviz' },
      target: { type: String, doc: 'format of the rendering image, e.g. svg' },
      alt: {type: String},
      width: {type: String},
      align: {type: String},
    },
    run(data, vfile2) {
      const sourceDocName = vfile2.history[vfile2.history.length - 1];
      const filename = data.arg;
      const rawText = 
        filename === undefined ? 
        data.body : 
        fs.readFileSync(
          path.join(path.dirname(sourceDocName), filename), 'utf8'
        );
      const postText = compress_b64(rawText);
      const src = data.options.src || 'graphviz';
      const target = data.options.target || 'svg';
      const image = { 
        type: 'image',
        url: `https://kroki.io/${src}/${target}/${postText}`,
        alt: data.options.alt,
        width: data.options.width,
        align: data.options.align,
      };
      return [image];
    },
  };
  
  const plugin = { name: 'kroki Images', directives: [krokiDirective] };
  
  export default plugin;
  
