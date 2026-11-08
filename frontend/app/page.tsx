import Workspace from "../components/Workspace";

export default function Page() {
  return <Workspace docsUrl={process.env.ALPHADRIVER_DOCS_URL ?? "http://127.0.0.1:8000/docs"} />;
}
