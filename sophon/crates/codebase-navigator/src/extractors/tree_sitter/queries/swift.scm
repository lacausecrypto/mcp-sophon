; Swift tag query. Modeled after the grammar's own tags.scm but
; trimmed to the declarations Sophon emits as top-level symbols.

(class_declaration
  name: (type_identifier) @name) @class

(protocol_declaration
  name: (type_identifier) @name) @interface

(function_declaration
  name: (simple_identifier) @name) @function

(protocol_function_declaration
  name: (simple_identifier) @name) @method

(typealias_declaration
  name: (type_identifier) @name) @type
