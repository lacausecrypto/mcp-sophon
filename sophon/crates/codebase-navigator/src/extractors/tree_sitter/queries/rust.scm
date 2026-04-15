; Tag query for Rust — captures top-level declarations via structural
; anchoring. Same reasoning as `python.scm`: tree-sitter core has no
; `#not-has-ancestor?` predicate, so we match explicitly on the parent
; node kind (source_file / declaration_list).
;
; - Free functions   = function_item direct child of source_file
; - Methods          = function_item directly inside an impl_item body
; - Types/consts     = same pattern, under source_file

(source_file
  (function_item
    name: (identifier) @name) @function)

; Methods: function_item inside an impl block's declaration_list.
(impl_item
  body: (declaration_list
    (function_item
      name: (identifier) @name) @method))

; Associated functions inside impl that aren't methods (no `self`)
; are still tagged as @method for navigation purposes — the ranker
; doesn't care about the self distinction.

(source_file
  (struct_item
    name: (type_identifier) @name) @struct)

(source_file
  (enum_item
    name: (type_identifier) @name) @enum)

(source_file
  (trait_item
    name: (type_identifier) @name) @trait)

(source_file
  (type_item
    name: (type_identifier) @name) @type)

(source_file
  (const_item
    name: (identifier) @name) @const)

(source_file
  (static_item
    name: (identifier) @name) @const)

(source_file
  (mod_item
    name: (identifier) @name) @module)
