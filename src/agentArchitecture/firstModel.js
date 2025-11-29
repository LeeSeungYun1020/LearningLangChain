import {DuckDuckGoSearch} from "@langchain/community/tools/duckduckgo_search";
import {Calculator} from "@langchain/community/tools/calculator";
import {ChatGoogleGenerativeAI} from "@langchain/google-genai";
import {
  Annotation,
  messagesStateReducer,
  START,
  StateGraph
} from "@langchain/langgraph";
import {ToolNode, toolsCondition} from "@langchain/langgraph/prebuilt";
import {AIMessage, HumanMessage} from "@langchain/core/messages";

const search = new DuckDuckGoSearch()
const calculator = new Calculator()
const tools = [search, calculator]
const model = new ChatGoogleGenerativeAI(
    {model: "gemini-2.5-flash-lite", temperature: 0.1}).bindTools(tools)
const annotation = Annotation.Root({
  messages: Annotation({
    reducer: messagesStateReducer, default: () => []
  })
})

async function firstModelNode(state) {
  const query = state.messages[state.messages.length - 1].content
  const searchToolCall = {
    name: search.name,
    args: {input: query},
    id: Math.random().toString(),
  }
  return {messages: [new AIMessage({ content: "", tool_calls: [searchToolCall]})]}
}

async function modelNode(state) {
  const res = await model.invoke(state.messages)
  return {messages: res}
}

// 검색 실패하는 경우 처리 추가해야 함
// 덕덕고가 Error: DDG detected an anomaly in the request, you are likely making requests too quickly.\n Please fix your mistakes.로 지속 실패하는 경우
// 계속해서 AI 호출하게 되는 이슈 있음
const builder = new StateGraph(annotation)
.addNode('first_model', firstModelNode)
.addNode('model', modelNode)
.addNode('tools', new ToolNode(tools))
.addEdge(START, 'first_model')
.addEdge('first_model', 'tools')
.addEdge('tools', 'model')
.addConditionalEdges('model', toolsCondition)
const graph = builder.compile()

const input = {
  messages: [
    new HumanMessage("대한민국의 제13대 대통령이 사망했을 때 몇 살이었나요?"), // 89세
  ],
}
for await (const chunk of await graph.stream(input)) {
  console.log(chunk)
}