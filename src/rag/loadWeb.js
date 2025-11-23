import {CheerioWebBaseLoader} from "@langchain/community/document_loaders/web/cheerio";
import {RecursiveCharacterTextSplitter} from "@langchain/textsplitters";
import {GoogleGenerativeAIEmbeddings} from "@langchain/google-genai";

// 문서 가져오기
const loader = new CheerioWebBaseLoader("https://leeseungyun1020.github.io/about")
const docs = await loader.load()

// 문서 분할
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
})
const chunks = await splitter.splitDocuments(docs)

const trimmedChunks = chunks.map(d => d.pageContent.split("\n").map(str => str.trim()).filter(l => l.length > 0).join("\n"))
console.log(trimmedChunks)

// 임베딩 생성
const model = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001", // 768 dimensions
  taskType: "RETRIEVAL_DOCUMENT",
  title: "about",
});
const embeddings = await model.embedDocuments(trimmedChunks)
console.log(embeddings)