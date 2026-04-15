; PHP tag query. The grammar exposes two languages: `language_php`
; (full PHP including HTML interleaving) and `language_php_only`
; (pure PHP, no `<?php` tags). We use `language_php` so bare `.php`
; files and framework templates parse the same way.

(namespace_definition
  name: (namespace_name) @name) @module

(class_declaration
  name: (name) @name) @class

(interface_declaration
  name: (name) @name) @interface

(trait_declaration
  name: (name) @name) @interface

(enum_declaration
  name: (name) @name) @enum

(function_definition
  name: (name) @name) @function

(method_declaration
  name: (name) @name) @method
