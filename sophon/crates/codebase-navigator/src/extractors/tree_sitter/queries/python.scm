; Python tag query.
;
; Uses structural anchoring: we match functions that are *direct*
; children of a `module` node (top-level) or inside a class body
; (methods). Nested inner functions — `def inner` inside `def outer` —
; are therefore naturally excluded without needing a custom
; `#not-has-ancestor?` predicate, which tree-sitter core does not ship.

(class_definition
  name: (identifier) @name) @class

; Methods: function_definition nested directly in a class body.
(class_definition
  body: (block
    (function_definition
      name: (identifier) @name) @method))

; Top-level functions: direct children of the `module` root.
(module
  (function_definition
    name: (identifier) @name) @function)

; Also handle the common case of a top-level function guarded by a
; decorator — tree-sitter wraps it in a `decorated_definition`, still
; at the module level.
(module
  (decorated_definition
    definition: (function_definition
      name: (identifier) @name)) @function)
