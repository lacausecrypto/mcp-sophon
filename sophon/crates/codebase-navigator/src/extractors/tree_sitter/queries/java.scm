; Java tag query. Tree-sitter Java wraps everything in a `program`
; node which contains `package_declaration`, `import_declaration`,
; and top-level `class_declaration` / `interface_declaration` /
; `enum_declaration` / `record_declaration`.
;
; Method declarations live inside each type's body.

(class_declaration
  name: (identifier) @name) @class

(interface_declaration
  name: (identifier) @name) @interface

(enum_declaration
  name: (identifier) @name) @enum

(record_declaration
  name: (identifier) @name) @struct

; Methods inside a class body
(class_declaration
  body: (class_body
    (method_declaration
      name: (identifier) @name) @method))

; Methods inside an interface body
(interface_declaration
  body: (interface_body
    (method_declaration
      name: (identifier) @name) @method))
