; Kotlin tag query for `tree-sitter-kotlin-ng`. The grammar uses a
; named `name:` field on every declaration, pointing at the plain
; `identifier` node (no separate type_identifier / simple_identifier
; distinction).

(class_declaration
  name: (identifier) @name) @class

(object_declaration
  name: (identifier) @name) @class

(function_declaration
  name: (identifier) @name) @function

(type_alias
  type: (identifier) @name) @type
