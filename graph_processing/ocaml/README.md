# Compile
ocamlopt str.cmxa translations.ml -o translations.native

# Use

./translations.native alpha beta gamma initial_vertex < graph_file > output_file
