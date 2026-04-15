; Ruby tag query.
;
; Structural anchoring to `program` for top-level, to `class`/`module`
; bodies for methods. No `#not-has-ancestor?` predicate needed.

; Top-level classes and modules (direct children of program)
(program
  (class
    name: [(constant) (scope_resolution)] @name) @class)

(program
  (module
    name: [(constant) (scope_resolution)] @name) @module)

; Methods inside a class body
(class
  body: (body_statement
    (method
      name: [(identifier) (constant)] @name) @method))

(class
  body: (body_statement
    (singleton_method
      name: [(identifier) (constant)] @name) @method))

; Methods inside a module body
(module
  body: (body_statement
    (method
      name: [(identifier) (constant)] @name) @method))

(module
  body: (body_statement
    (singleton_method
      name: [(identifier) (constant)] @name) @method))

; Top-level functions (method nodes that are direct children of program)
(program
  (method
    name: [(identifier) (constant)] @name) @function)
