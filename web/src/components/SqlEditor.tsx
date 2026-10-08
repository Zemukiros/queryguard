/**
 * CodeMirror 6, PostgreSQL dialect. Colours come from the CSS tokens, so the
 * editor follows the light/dark theme without being rebuilt.
 */
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { PostgreSQL, sql } from "@codemirror/lang-sql";
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { EditorState } from "@codemirror/state";
import { EditorView, keymap, lineNumbers, placeholder as placeholderExt } from "@codemirror/view";
import { tags as t } from "@lezer/highlight";
import { useEffect, useRef } from "react";

const highlight = HighlightStyle.define([
  { tag: [t.keyword, t.operatorKeyword, t.modifier], color: "var(--code-keyword)", fontWeight: "600" },
  { tag: [t.string, t.special(t.string)], color: "var(--code-string)" },
  { tag: [t.number, t.bool, t.null], color: "var(--code-number)" },
  { tag: [t.comment, t.lineComment, t.blockComment], color: "var(--code-comment)", fontStyle: "italic" },
  { tag: [t.function(t.variableName), t.standard(t.name)], color: "var(--code-fn)" },
  { tag: [t.operator, t.punctuation], color: "var(--code-op)" },
  { tag: [t.typeName], color: "var(--code-fn)" },
]);

const theme = EditorView.theme({
  "&": { backgroundColor: "var(--surface-2)", color: "var(--ink)", borderRadius: "6px" },
  "&.cm-focused": { outline: "2px solid var(--accent)", outlineOffset: "1px" },
  ".cm-content": { padding: "8px 0", caretColor: "var(--ink)" },
  ".cm-gutters": { backgroundColor: "transparent", color: "var(--ink-3)", border: "none" },
  ".cm-activeLine, .cm-activeLineGutter": { backgroundColor: "transparent" },
  ".cm-selectionBackground, &.cm-focused .cm-selectionBackground, ::selection": { backgroundColor: "var(--accent-soft)" },
  ".cm-placeholder": { color: "var(--ink-3)" },
  ".cm-scroller": { maxHeight: "260px", overflow: "auto" },
});

export function SqlEditor({ value, onChange, readOnly = false, label, placeholder, testId }: {
  value: string;
  onChange?: (value: string) => void;
  readOnly?: boolean;
  label: string;
  placeholder?: string;
  testId?: string;
}) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const change = useRef(onChange);
  const initial = useRef(value);

  useEffect(() => {
    change.current = onChange;
  }, [onChange]);

  useEffect(() => {
    if (!host.current) return;
    const editor = new EditorView({
      parent: host.current,
      state: EditorState.create({
        doc: initial.current,
        extensions: [
          lineNumbers(),
          history(),
          keymap.of([...defaultKeymap, ...historyKeymap]),
          sql({ dialect: PostgreSQL }),
          syntaxHighlighting(highlight),
          theme,
          EditorView.lineWrapping,
          EditorState.readOnly.of(readOnly),
          EditorView.editable.of(!readOnly),
          EditorView.contentAttributes.of({ "aria-label": label, ...(testId ? { "data-testid": testId } : {}) }),
          placeholderExt(placeholder ?? ""),
          EditorView.updateListener.of((update) => {
            if (update.docChanged) change.current?.(update.state.doc.toString());
          }),
        ],
      }),
    });
    view.current = editor;
    return () => {
      editor.destroy();
      view.current = null;
    };
  }, [readOnly, label, placeholder, testId]);

  // Follow value changes made outside the editor (a new answer, an example).
  useEffect(() => {
    const editor = view.current;
    if (!editor) return;
    const current = editor.state.doc.toString();
    if (current !== value) editor.dispatch({ changes: { from: 0, to: current.length, insert: value } });
  }, [value]);

  return <div ref={host} />;
}
