; JavaScript tag query.

(function_declaration
  name: (identifier) @name) @function

(class_declaration
  name: (identifier) @name) @class

; `const foo = (…) => …` — a VariableDeclarator whose value is an
; arrow function. We only tag the name.
(lexical_declaration
  (variable_declarator
    name: (identifier) @name
    value: [(arrow_function) (function_expression)])) @function

(variable_declaration
  (variable_declarator
    name: (identifier) @name
    value: [(arrow_function) (function_expression)])) @function
