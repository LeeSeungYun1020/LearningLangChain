import {
  Annotation,
  END,
  MemorySaver,
  messagesStateReducer,
  START,
  StateGraph,
} from '@langchain/langgraph'
import {ChatGoogleGenerativeAI} from "@langchain/google-genai";
import * as fs from "node:fs";
import {HumanMessage} from "@langchain/core/messages";

const model = new ChatGoogleGenerativeAI({model: "gemini-2.5-flash-lite"})

// 상태 정의
const state = {
  messages: Annotation({
    reducer: messagesStateReducer, default: () => [],
  }),
}

// 그래프 구성
let graphBuilder = new StateGraph(state)
async function chatbot(state) {
  const answer = await model.invoke(state.messages)
  return {messages: answer}
}
graphBuilder = graphBuilder.addNode('chatbot', chatbot).addEdge(START, 'chatbot').addEdge('chatbot', END)

// 그래프 생성 및 이미지로 저장
let graph = graphBuilder.compile({checkpointer: new MemorySaver()})
fs.writeFileSync('graph.png', new Uint8Array(
    await (await (await graph.getGraphAsync()).drawMermaidPng()).arrayBuffer()))

// 스레드 생성 - 그래프에서 스레드로 사용자 구분 가능
const thread = {configurable: {thread_id: '1'}}
const result1 = await graph.invoke(
    {messages: [new HumanMessage("Hi there! I am yun")]}, thread)
const result2 = await graph.invoke(
    {messages: [new HumanMessage("What is my name??")]}, thread)
console.log(result1, result2)
const input = {messages: [new HumanMessage("It's fine. Thanks!")]}
for await (const chunk of await graph.stream(input, thread)) {
  console.log(chunk)
}