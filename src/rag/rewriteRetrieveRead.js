import {ChatPromptTemplate} from "@langchain/core/prompts";
import {RunnableLambda} from "@langchain/core/runnables";
import {initChatModel} from "langchain";
import {
  CheerioWebBaseLoader
} from "@langchain/community/document_loaders/web/cheerio";
import {RecursiveCharacterTextSplitter} from "@langchain/textsplitters";
import {MemoryVectorStore} from "@langchain/classic/vectorstores/memory";
import {GoogleGenerativeAIEmbeddings} from "@langchain/google-genai";

export async function loadWeb(url) {
  // 문서 가져오기
  const loader = new CheerioWebBaseLoader(url)
  const docs = await loader.load()

  // 문서 분할
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  })
  return await splitter.splitDocuments(docs)
}

// 임베딩 생성 및 저장
const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "text-embedding-004", // 768 dimensions
  taskType: "RETRIEVAL_DOCUMENT",
  title: "Document",
})
const document = await loadWeb("https://namu.wiki/w/%EC%88%98%EB%8F%84(%EB%8F%84%EC%8B%9C)")
const vectorStore = await MemoryVectorStore.fromDocuments(document, embeddings)

// 검색
const retriever = vectorStore.asRetriever()
const prompt = ChatPromptTemplate.fromTemplate(
    "다음 컨텍스트를 이용하여 질문에 답변해줘."
    + "컨텍스트: {context}"
    + "질문: {question}"
)

// 재작성
const model = await initChatModel("google-genai:gemini-2.5-flash-lite")
const rewritePrompt = ChatPromptTemplate.fromTemplate(
    '웹 검색 엔진이 다음 질문 내용에 답을 찾을 수 있도록 검색어를 한 문장으로 제공해줘.\n\n질문: {question}\n검색어: '
)
const rewriter = rewritePrompt.pipe(model).pipe((message) => {
  console.log("Rewriter: " + message.content)
  return message.content.replaceAll('"', '').replaceAll("**", "").replaceAll("검색어: ", "")
})

const rewriteQA = RunnableLambda.from(async (input) => {
  const newQuery = await rewriter.invoke({question: input})
  const docs = await retriever.invoke(newQuery)
  console.log(docs.map(d => d.pageContent))
  const formatted = await prompt.invoke({context: docs, question: input})
  return await model.invoke(formatted)
})

const result = await rewriteQA.invoke(
    "아이 추워. 어제 아침에 전화가 와서 받았는데, 친구가 일본에 갔다는거야."
    + "교토에 갔는데 거기가 옛날에 수도였다고 해."
    + "내가 우리나라도 예전엔 수도가 서울이 아니였을 수도 있을 것 같다고 하니까 걔가 거짓말하지 말라고 하는거야."
    + "예전에 수도였던 곳을 알려주면 친구에게 설명할 수 있을 것 같아."
)
console.log(result)

// 한글은 잘 못 찾음 / 벡터 스토어의 한계로 다음과 같이 직접 검색해도 역대 한국의 수도가 포함된 문단을 찾아내지 못하는 것 같음
// console.log(await retriever.invoke("역대 한국의 수도"))