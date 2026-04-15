; Go tag query.

(function_declaration
  name: (identifier) @name) @function

(method_declaration
  name: (field_identifier) @name) @method

(type_declaration
  (type_spec
    name: (type_identifier) @name
    type: (struct_type))) @struct

(type_declaration
  (type_spec
    name: (type_identifier) @name
    type: (interface_type))) @interface

; Generic `type Foo = Bar` or `type Foo Bar` (not struct/interface)
; would also match the two rules above via `type_spec` — tree-sitter
; rules are OR'd but the captures keep their order, so struct and
; interface win when present. Remaining type aliases fall through.
(type_declaration
  (type_spec
    name: (type_identifier) @name)) @type
