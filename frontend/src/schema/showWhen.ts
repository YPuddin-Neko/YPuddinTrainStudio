/**
 * Tiny expression language for conditional field visibility (`x-ui.show_when`).
 * Ported 1:1 from `ypuddin/config/show_when.py`.
 *
 * Grammar (precedence low -> high):
 *   expr    := or
 *   or      := and ("||" and)*
 *   and     := not ("&&" not)*
 *   not     := "!" not | cmp
 *   cmp     := atom (("==" | "!=" | "<" | "<=" | ">" | ">=" | "in") atom)?
 *   atom    := path | string | number | "true" | "false" | "null" | "[" (atom ("," atom)*)? "]" | "(" expr ")"
 *   path    := ident ("." ident)*
 */

export class ShowWhenError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ShowWhenError';
  }
}

type TokKind = 'num' | 'str' | 'op' | 'bool' | 'null' | 'path';

interface Tok {
  kind: TokKind;
  value: any;
}

// Tokenizer regex matching numbers, single/double quoted strings, operators, and identifiers/paths
const TOKEN_RE = /\s*(?:(-?\d+(?:\.\d+)?)|('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")|(==|!=|<=|>=|&&|\|\||[<>!()[\],])|([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*))/y;

export function tokenize(src: string): Tok[] {
  let pos = 0;
  const out: Tok[] = [];
  const trimmed = src.trim();
  TOKEN_RE.lastIndex = 0;

  while (pos < trimmed.length) {
    TOKEN_RE.lastIndex = pos;
    const m = TOKEN_RE.exec(trimmed);
    if (!m || TOKEN_RE.lastIndex === pos) {
      throw new ShowWhenError(`bad token at ${pos}: ${JSON.stringify(trimmed.slice(pos, pos + 10))}`);
    }
    pos = TOKEN_RE.lastIndex;

    const [, num, str, op, path] = m;
    if (num !== undefined) {
      out.push({ kind: 'num', value: num.includes('.') ? parseFloat(num) : parseInt(num, 10) });
    } else if (str !== undefined) {
      const raw = str.slice(1, -1);
      // Unescape standard escape sequences
      const unescaped = raw.replace(/\\(['"\\])/g, '$1').replace(/\\n/g, '\n').replace(/\\t/g, '\t');
      out.push({ kind: 'str', value: unescaped });
    } else if (op !== undefined) {
      out.push({ kind: 'op', value: op });
    } else if (path !== undefined) {
      if (path === 'true' || path === 'false') {
        out.push({ kind: 'bool', value: path === 'true' });
      } else if (path === 'null') {
        out.push({ kind: 'null', value: null });
      } else if (path === 'in') {
        out.push({ kind: 'op', value: 'in' });
      } else {
        out.push({ kind: 'path', value: path });
      }
    }
  }
  return out;
}

export type ASTNode =
  | { type: 'lit'; value: any }
  | { type: 'path'; value: string }
  | { type: 'list'; items: ASTNode[] }
  | { type: 'not'; expr: ASTNode }
  | { type: 'and'; left: ASTNode; right: ASTNode }
  | { type: 'or'; left: ASTNode; right: ASTNode }
  | { type: 'cmp'; op: string; left: ASTNode; right: ASTNode };

class Parser {
  private toks: Tok[];
  private i = 0;

  constructor(tokens: Tok[]) {
    this.toks = tokens;
  }

  peek(): Tok | null {
    return this.i < this.toks.length ? this.toks[this.i] : null;
  }

  take(kind?: TokKind, value?: any): Tok {
    const tok = this.peek();
    if (!tok || (kind && tok.kind !== kind) || (value !== undefined && tok.value !== value)) {
      throw new ShowWhenError(`expected ${value || kind}, got ${tok ? JSON.stringify(tok) : 'EOF'}`);
    }
    this.i++;
    return tok;
  }

  parse(): ASTNode {
    const node = this.pOr();
    if (this.peek() !== null) {
      throw new ShowWhenError(`trailing tokens: ${JSON.stringify(this.toks.slice(this.i))}`);
    }
    return node;
  }

  private pOr(): ASTNode {
    let node = this.pAnd();
    let t = this.peek();
    while (t && t.kind === 'op' && t.value === '||') {
      this.take();
      node = { type: 'or', left: node, right: this.pAnd() };
      t = this.peek();
    }
    return node;
  }

  private pAnd(): ASTNode {
    let node = this.pNot();
    let t = this.peek();
    while (t && t.kind === 'op' && t.value === '&&') {
      this.take();
      node = { type: 'and', left: node, right: this.pNot() };
      t = this.peek();
    }
    return node;
  }

  private pNot(): ASTNode {
    const t = this.peek();
    if (t && t.kind === 'op' && t.value === '!') {
      this.take();
      return { type: 'not', expr: this.pNot() };
    }
    return this.pCmp();
  }

  private pCmp(): ASTNode {
    const left = this.pAtom();
    const t = this.peek();
    if (t && t.kind === 'op' && ['==', '!=', '<', '<=', '>', '>=', 'in'].includes(t.value)) {
      this.take();
      const right = this.pAtom();
      return { type: 'cmp', op: t.value, left, right };
    }
    return left;
  }

  private pAtom(): ASTNode {
    const t = this.take();
    if (t.kind === 'op' && t.value === '(') {
      const node = this.pOr();
      this.take('op', ')');
      return node;
    }
    if (t.kind === 'op' && t.value === '[') {
      const items: ASTNode[] = [];
      let n = this.peek();
      while (!(n && n.kind === 'op' && n.value === ']')) {
        items.push(this.pAtom());
        n = this.peek();
        if (n && n.kind === 'op' && n.value === ',') {
          this.take();
          n = this.peek();
        }
      }
      this.take('op', ']');
      return { type: 'list', items };
    }
    if (t.kind === 'num' || t.kind === 'str' || t.kind === 'bool' || t.kind === 'null') {
      return { type: 'lit', value: t.value };
    }
    if (t.kind === 'path') {
      return { type: 'path', value: t.value };
    }
    throw new ShowWhenError(`unexpected token ${JSON.stringify(t)}`);
  }
}

export function parse(expr: string): ASTNode {
  return new Parser(tokenize(expr)).parse();
}

function lookup(data: any, path: string): any {
  let cur = data;
  for (const part of path.split('.')) {
    if (cur === null || cur === undefined) {
      return null;
    }
    if (typeof cur === 'object') {
      cur = cur[part];
    } else {
      return null;
    }
    if (cur === undefined) {
      return null;
    }
  }
  return cur === undefined ? null : cur;
}

function evalNode(node: ASTNode, data: any): any {
  if (node.type === 'lit') {
    return node.value;
  }
  if (node.type === 'path') {
    return lookup(data, node.value);
  }
  if (node.type === 'list') {
    return node.items.map((n) => evalNode(n, data));
  }
  if (node.type === 'not') {
    return !evalNode(node.expr, data);
  }
  if (node.type === 'and') {
    return Boolean(evalNode(node.left, data)) && Boolean(evalNode(node.right, data));
  }
  if (node.type === 'or') {
    return Boolean(evalNode(node.left, data)) || Boolean(evalNode(node.right, data));
  }
  if (node.type === 'cmp') {
    const left = evalNode(node.left, data);
    const right = evalNode(node.right, data);
    const op = node.op;

    if (op === '==') {
      return left === right;
    }
    if (op === '!=') {
      return left !== right;
    }
    if (op === 'in') {
      if (Array.isArray(right)) {
        return right.includes(left);
      }
      if (typeof right === 'string') {
        return right.includes(String(left));
      }
      return false;
    }
    // Inequality comparisons: if either side is null/undefined, return false
    if (left === null || left === undefined || right === null || right === undefined) {
      return false;
    }
    if (op === '<') return left < right;
    if (op === '<=') return left <= right;
    if (op === '>') return left > right;
    if (op === '>=') return left >= right;
  }
  throw new ShowWhenError(`bad node ${JSON.stringify(node)}`);
}

/**
 * Evaluate `expr` against a config object. Missing paths evaluate as `null`.
 * Throws `ShowWhenError` on syntax or parse errors.
 */
export function evaluateShowWhen(expr: string, data: any): boolean {
  if (!expr || typeof expr !== 'string' || !expr.trim()) return true;
  return Boolean(evalNode(parse(expr), data));
}
