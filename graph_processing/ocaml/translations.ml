let alpha = try float_of_string Sys.argv.(1) with _ -> 1.
let beta = try float_of_string Sys.argv.(2) with _ -> 1.
let gamma = try float_of_string Sys.argv.(3) with _ -> 1.

let _ = Printf.printf "Command is: %s alpha beta gamma initial_vertex\nComputing for alpha = %f, beta = %f and gamma = %f\n%!" Sys.argv.(0) alpha beta gamma

let grid_graph n =
  Array.init
    (n*n)
    (fun i ->
      let res = ref [] in
      if i mod n > 0
      then res := (i-1) :: !res;
      if i mod n < n-1
      then res := (i+1) :: !res;
      if i > n - 1
      then res := (i-n) :: !res;
      if i < n*n - n
      then res := (i+n) :: !res;
      !res
    )
  
let graph = grid_graph 6
          
let graph =
  let res = Array.init (int_of_string (input_line stdin)) (fun _ -> List.map int_of_string (Str.split (Str.regexp " ") (input_line stdin))) in  
  res

let initial_vertex = try int_of_string Sys.argv.(4) with _ -> Random.self_init (); Random.int (Array.length graph)
let _ = Printf.eprintf "Initial vertex is %d\n%!" initial_vertex      

let compute_score initial_graph graph index translation l gts stg =
  let loss_score = ref 0 in
  for i = 0 to index do
    if translation.(i) = -1
    then incr loss_score;
  done;  
  let injection_score = ref 0 in
  for i = 0 to index do
    for j = 0 to i - 1 do
      if translation.(i) = translation.(j) && translation.(i) <> (-1)
      then incr injection_score
    done;
  done;
  let edge_constrained_score = ref 0 in
  for i = 0 to index do
    if not (List.mem translation.(i) graph.(stg.(i))) && translation.(i) <> (-1)
    then incr edge_constrained_score
  done;
  let snp_score = ref 0 in
  for i = 0 to index do
    if translation.(i) <> (-1)
    then 
      for j = 0 to index do
        if translation.(j) <> (-1)
        then 
          try
            if List.mem stg.(i) initial_graph.(stg.(j)) && not (List.mem translation.(i) initial_graph.(translation.(j)))
            then incr snp_score;
            if not (List.mem stg.(i) initial_graph.(stg.(j))) && List.mem translation.(i) initial_graph.(translation.(j))
            then incr snp_score
          with _ -> ()
      done;
  done;
  let foi = float_of_int in
  foi !loss_score +. alpha *. foi !injection_score +. beta *. foi !edge_constrained_score +. gamma *. foi !snp_score +. if not (List.mem translation.(0) graph.(stg.(0))) then 1000000. else 0.

let search initial_graph graph i l gts stg=
  let best_score = ref infinity in
  let best_translation = ref [||] in
  let rec search score index translation =
    if index = l
    then
      begin
        if score < !best_score
        then begin best_score := score; best_translation := Array.copy translation end;
      end
    else
      for image = -1 to Array.length stg - 1 do       
        translation.(index) <- if image <> -1 then stg.(image) else -1;
        let new_score = compute_score initial_graph graph index translation l gts stg in
        if new_score < !best_score
        then search new_score (index + 1) translation         
      done    
  in
  search 0. 0 (Array.make l (-1));
  !best_score, !best_translation

let induced_subgraph graph central_vertex =
  let list = central_vertex :: graph.(central_vertex) in
  let l = List.length list in
  let graph_to_subgraph = Hashtbl.create 100 in
  let subgraph_to_graph = ref [] in
  let count = ref 0 in
  List.iter (fun i -> Hashtbl.add graph_to_subgraph i !count; subgraph_to_graph := !subgraph_to_graph @ [i]; incr(count)) list;
  List.iter (fun i -> List.iter (fun j -> if not (List.mem j !subgraph_to_graph) then begin Hashtbl.add graph_to_subgraph j !count; subgraph_to_graph := !subgraph_to_graph @ [j]; incr(count) end) graph.(i)) list;
  l, graph_to_subgraph, Array.of_list (!subgraph_to_graph)

let identify_local_translations graph =
  let res = Array.make (Array.length graph) [] in
  let tables = Array.init (Array.length graph) (fun i -> induced_subgraph graph i) in
  for i = 0 to Array.length graph - 1 do
    Printf.eprintf "%d\n%!" i;
    let l, gts, stg = tables.(i) in
    let graph_copy = Array.copy graph in
    for j = 1 to l - 1 do
      let best_cost, best_translation = search graph graph_copy i l gts stg in
      res.(i) <- res.(i) @ [best_cost, best_translation];
      for i = 0 to Array.length best_translation - 1 do
        graph_copy.(stg.(i)) <- List.filter (fun x -> x <> best_translation.(i)) graph_copy.(stg.(i))
      done;
    done;
  done;
  res, tables

let local_translations, tables = identify_local_translations graph

let propagate_central_pattern graph central_vertex local_translations tables =
  let l, gts, stg = induced_subgraph graph central_vertex in
  let patterns = Array.init (Array.length graph) (fun _ -> (Array.make l (-1))) in
  let costs = Array.init (Array.length graph) (fun _ -> 10000.) in
  for i = 0 to l - 1 do
    patterns.(central_vertex).(i) <- stg.(i)
  done;
  costs.(central_vertex) <- 0.;
  
  begin
    try
      while true do
        Printf.eprintf "%f\n%!" (Array.fold_left (fun a b -> a +. b) 0. costs);
        let modifications = ref false in        
        for line = 0 to Array.length graph - 1 do
          let _,gts,_ = tables.(line) in
          List.iter
            (fun (cost, translation) ->
              let new_cost = cost +. costs.(line) in
              
              if new_cost < costs.(translation.(0))
              then
                begin
                  Array.iteri
                    (fun reference assignation ->
                      try
                        let new_assignation = Hashtbl.find gts assignation in
                        patterns.(translation.(0)).(reference) <- try translation.(new_assignation) with _ -> -1
                      with Not_found -> ()
                    ) (patterns.(line));
                  modifications := true;
                  costs.(translation.(0)) <- new_cost
                end
            ) local_translations.(line);
        done;
        if not (!modifications)
        then failwith "finished"
      done
    with Failure "finished" -> ()
  end;
  patterns, costs
  
let patterns, costs = propagate_central_pattern graph initial_vertex local_translations tables

let _ =
  Printf.printf "%d\n%!" (Array.length patterns);
  Array.iter
    (fun pattern ->
      Array.iteri
        (fun i x ->
          if x <> (-1)
          then Printf.printf "%s%d:%d" (if i = 0 then "" else " ") i x;           
        ) pattern;
      print_newline ();
    ) patterns
      
