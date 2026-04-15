; C++ tag query (works for both C and C++ sources — C++ grammar is a
; superset that also accepts plain C).

(function_definition
  declarator: (function_declarator
    declarator: (identifier) @name)) @function

(function_definition
  declarator: (function_declarator
    declarator: (qualified_identifier
      name: (identifier) @name))) @function

(declaration
  declarator: (function_declarator
    declarator: (identifier) @name)) @function

(class_specifier
  name: (type_identifier) @name) @class

(struct_specifier
  name: (type_identifier) @name) @struct

(union_specifier
  name: (type_identifier) @name) @struct

(enum_specifier
  name: (type_identifier) @name) @enum

(namespace_definition
  name: (namespace_identifier) @name) @module

(alias_declaration
  name: (type_identifier) @name) @type

(type_definition
  declarator: (type_identifier) @name) @type
