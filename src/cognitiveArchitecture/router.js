import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings
} from "@langchain/google-genai";
import {
  Annotation, END,
  messagesStateReducer, START,
  StateGraph
} from "@langchain/langgraph";
import {MemoryVectorStore} from "@langchain/classic/vectorstores/memory";
import {HumanMessage, SystemMessage} from "@langchain/core/messages";

const embeddings = new GoogleGenerativeAIEmbeddings()
const queryModel = new ChatGoogleGenerativeAI(
    {model: "gemini-2.5-flash-lite", temperature: 0.1})
const naturalModel = new ChatGoogleGenerativeAI(
    {model: "gemini-2.5-flash-lite", temperature: 0.7})
const annotation = Annotation.Root({
  messages: Annotation({reducer: messagesStateReducer, default: () => []}),
  user_query: Annotation(),
  domain: Annotation(),
  documents: Annotation(),
  answer: Annotation(),
})
const medicalRecordsStore = await MemoryVectorStore.fromDocuments([],
    embeddings)
const medicalRecordsRetriever = medicalRecordsStore.asRetriever()
const insuranceFaqsStore = await MemoryVectorStore.fromDocuments([], embeddings)
const insuranceFaqsRetriever = insuranceFaqsStore.asRetriever()

const routerPrompt = new SystemMessage(`
 사용자 문의를 어떤 도메인으로 라우팅할지 결정하여 도메인 이름만 출력해.
 - records: 환자 의료 기록. 진단, 치료, 처방 정보 포함
 - insurance: 보험 정책, 청구, 보장 관련 자주 묻는 질문.
`)

async function routerNode(state) {
  const userMessage = new HumanMessage(state.user_query)
  const messages = [routerPrompt, ...state.messages, userMessage]
  const res = await queryModel.invoke(messages)
  return {
    domain: res.content,
    messages: [userMessage, res]
  }
}

function pickRetriever(state) {
  if (state.domain === "records") {
    return 'retrieveMedicalRecords'
  } else if (state.domain === "insurance") {
    return 'retrieveInsuranceFaqs'
  } else {
    return 'error'
  }
}

async function retrieveMedicalRecords(state) {
  //const docs = await medicalRecordsRetriever.invoke(state.user_query)
  // 메모리 임시 구성
  return {documents: "환자는 2022년 11월 29일에 코로나19로 진단받아 해열제와 진해거담제를 처방받았습니다."}
}

async function retrieveInsuranceFaqs(state) {
  // const docs = await insuranceFaqsRetriever.invoke(state.user_query)
  // 메모리 임시 구성
  return {documents: "우리 보험사에서는 2020년 이전에 가입한 고객에게는 약관에 의해 코로나 19 관련 질환에 대해 보장하지 않습니다."}
}

const medicalRecordsPrompt = new SystemMessage(
    `당신은 유능한 의사입니다. 환자 의료 기록에 포함된 진단, 치료, 처방 정보를 기반으로 질문에 답하세요.`)
const insuranceRecordsPrompt = new SystemMessage(
    `당신은 전문적인 의료 보험 챗봇입니다. 보험 정책, 청구, 보장 관련 질문에 답하세요.`)

async function generateAnswer(state) {
  let prompt
  if (state.domain === "records") {
    prompt = medicalRecordsPrompt
  } else if (state.domain === "insurance") {
    prompt = insuranceRecordsPrompt
  } else {
    return {answer: "죄송합니다. 답변할 수 없는 질문입니다.", messages: null}
  }
  const messages = [prompt, ...state.messages,
    new HumanMessage(`Documents: ${state.documents}`)]
  const res = await naturalModel.invoke(messages)
  return {answer: res.content, messages: res}
}

const builder = new StateGraph(annotation)
.addNode('router', routerNode)
.addNode('retrieveMedicalRecords', retrieveMedicalRecords)
.addNode('retrieveInsuranceFaqs', retrieveInsuranceFaqs)
.addNode('generateAnswer', generateAnswer)
.addEdge(START, 'router')
.addConditionalEdges('router', pickRetriever)
.addEdge('retrieveMedicalRecords', 'generateAnswer')
.addEdge('retrieveInsuranceFaqs', 'generateAnswer')
.addEdge('generateAnswer', END)
const graph = builder.compile()

// 실행

const input ={user_query: `코로나19는 보험 적용이 되나요?`}
for await (const chunk of await graph.stream(input)) {
  console.log(chunk)
}