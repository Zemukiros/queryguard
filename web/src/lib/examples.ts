export interface Example {
  label: string;
  question: string;
  /** Set: the example puts this SQL in the editor and runs it, instead of asking. */
  sql?: string;
}

export const EXAMPLES: Example[] = [
  { label: "How many orders were cancelled?", question: "How many orders were cancelled?" },
  { label: "Who are our top 10 customers? (ambiguous)", question: "Who are our top 10 customers?" },
  { label: "Paste a DROP TABLE (guardrail)", question: "Clean up the orders table", sql: "DROP TABLE orders;" },
];
