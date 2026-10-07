const GOOD="https://raw.githubusercontent.com/Belerafon/LLM_watermark_payground/910e62f93677496c3c4a3f5981202d4c8b7e8c7e/main.js";
let code=await(await fetch(GOOD)).text();
code=code.replaceAll("models.js?v=6","models.js?v=7").replaceAll("worker-v4.js?v=17","worker-v4.js?v=18");
const base=new URL("./",import.meta.url).href;
code=code.replaceAll('from"./','from"'+base).replaceAll('new Worker("','new Worker("'+base);
await import(URL.createObjectURL(new Blob([code],{type:"text/javascript"})));
