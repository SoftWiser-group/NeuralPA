[BugLab_Wrong_Literal]^this.counter = -1;^46^^^^^44^47^this.counter = 0;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] <init> [RETURN_TYPE] AbstractCompiler)   AbstractCompiler compiler [VARIABLES] AbstractCompiler  compiler  int  counter  boolean  
[BugLab_Variable_Misuse]^NodeTraversal.traverse ( compiler, externs, new Traversal (  )  ) ;^51^^^^^50^52^NodeTraversal.traverse ( compiler, root, new Traversal (  )  ) ;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] process [RETURN_TYPE] void   Node externs Node root [VARIABLES] AbstractCompiler  compiler  boolean  int  counter  Node  externs  root  
[BugLab_Argument_Swapping]^NodeTraversal.traverse ( root, compiler, new Traversal (  )  ) ;^51^^^^^50^52^NodeTraversal.traverse ( compiler, root, new Traversal (  )  ) ;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] process [RETURN_TYPE] void   Node externs Node root [VARIABLES] AbstractCompiler  compiler  boolean  int  counter  Node  externs  root  
[BugLab_Variable_Misuse]^if  ( tmp.getType (  )  == Token.FOR && n.getChildCount (  )  == 3 )  {^58^^^^^43^73^if  ( n.getType (  )  == Token.FOR && n.getChildCount (  )  == 3 )  {^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Wrong_Operator]^if  ( n.getType (  )  == Token.FOR || n.getChildCount (  )  == 3 )  {^58^^^^^43^73^if  ( n.getType (  )  == Token.FOR && n.getChildCount (  )  == 3 )  {^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Wrong_Operator]^if  ( n.getType (  )  <= Token.FOR && n.getChildCount (  )  == 3 )  {^58^^^^^43^73^if  ( n.getType (  )  == Token.FOR && n.getChildCount (  )  == 3 )  {^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Wrong_Operator]^if  ( n.getType (  )  == Token.FOR && n.getChildCount (  )  != 3 )  {^58^^^^^43^73^if  ( n.getType (  )  == Token.FOR && n.getChildCount (  )  == 3 )  {^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Wrong_Operator]^if  ( n.getType (  )  > Token.FOR && n.getChildCount (  )  == 3 )  {^58^^^^^43^73^if  ( n.getType (  )  == Token.FOR && n.getChildCount (  )  == 3 )  {^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Wrong_Literal]^if  ( n.getType (  )  == Token.FOR && n.getChildCount (  )  == counter )  {^58^^^^^43^73^if  ( n.getType (  )  == Token.FOR && n.getChildCount (  )  == 3 )  {^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^if  ( tmp.getType (  )  == Token.VAR )  {^70^^^^^55^85^if  ( key.getType (  )  == Token.VAR )  {^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Wrong_Operator]^if  ( key.getType (  )  <= Token.VAR )  {^70^^^^^55^85^if  ( key.getType (  )  == Token.VAR )  {^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^new Node ( Token.ASSIGN, newBody, tmp.cloneTree (  )  )  ) ,^103^104^105^106^^88^118^new Node ( Token.ASSIGN, key, tmp.cloneTree (  )  )  ) ,^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^new Node ( Token.ASSIGN, key, parent.cloneTree (  )  )  ) ,^103^104^105^106^^88^118^new Node ( Token.ASSIGN, key, tmp.cloneTree (  )  )  ) ,^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Argument_Swapping]^new Node ( Token.ASSIGN, tmp, key.cloneTree (  )  )  ) ,^103^104^105^106^^88^118^new Node ( Token.ASSIGN, key, tmp.cloneTree (  )  )  ) ,^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^ifBody = new Node ( Token.BLOCK, newBody, new Node (^80^81^82^83^^65^95^ifBody = new Node ( Token.BLOCK, key, new Node (^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^new Node ( Token.ASSIGN, parent.getFirstChild (  ) .cloneNode (  ) , tmp.cloneTree (  )  )  ) ,^85^86^87^88^^70^100^new Node ( Token.ASSIGN, key.getFirstChild (  ) .cloneNode (  ) , tmp.cloneTree (  )  )  ) ,^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^new Node ( Token.ASSIGN, key.getFirstChild (  ) .cloneNode (  ) , parent.cloneTree (  )  )  ) ,^85^86^87^88^^70^100^new Node ( Token.ASSIGN, key.getFirstChild (  ) .cloneNode (  ) , tmp.cloneTree (  )  )  ) ,^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Argument_Swapping]^new Node ( Token.ASSIGN, tmp.getFirstChild (  ) .cloneNode (  ) , key.cloneTree (  )  )  ) ,^85^86^87^88^^70^100^new Node ( Token.ASSIGN, key.getFirstChild (  ) .cloneNode (  ) , tmp.cloneTree (  )  )  ) ,^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^n.addChildToFront ( new Node ( Token.VAR, n )  ) ;^65^^^^^50^80^n.addChildToFront ( new Node ( Token.VAR, tmp )  ) ;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^ifBody = new Node ( Token.BLOCK, tmp, new Node (^80^81^82^83^^65^95^ifBody = new Node ( Token.BLOCK, key, new Node (^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^new Node ( Token.GETPROP, n.cloneTree (  ) , Node.newString ( "match" )  ) ,^119^120^121^122^^104^134^new Node ( Token.GETPROP, tmp.cloneTree (  ) , Node.newString ( "match" )  ) ,^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^Node body = tmp.getLastChild (  ) ;^59^^^^^44^74^Node body = n.getLastChild (  ) ;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^Node key = parent.getFirstChild (  ) ;^61^^^^^46^76^Node key = n.getFirstChild (  ) ;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^Node body = parent.getLastChild (  ) ;^59^^^^^44^74^Node body = n.getLastChild (  ) ;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^n.removeChild ( tmp ) ;^60^^^^^45^75^n.removeChild ( body ) ;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^Node key = tmp.getFirstChild (  ) ;^61^^^^^46^76^Node key = n.getFirstChild (  ) ;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^n.removeChild ( tmp ) ;^62^^^^^47^77^n.removeChild ( key ) ;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^n.addChildToFront ( parentew Node ( Token.VAR, tmp )  ) ;^65^^^^^50^80^n.addChildToFront ( new Node ( Token.VAR, tmp )  ) ;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^n.addChildToFront ( new Node ( Token.VAR, newBody )  ) ;^65^^^^^50^80^n.addChildToFront ( new Node ( Token.VAR, tmp )  ) ;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Argument_Swapping]^n.addChildToFront ( tmpew Node ( Token.VAR, n )  ) ;^65^^^^^50^80^n.addChildToFront ( new Node ( Token.VAR, tmp )  ) ;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^if  ( newBody.getType (  )  == Token.VAR )  {^70^^^^^55^85^if  ( key.getType (  )  == Token.VAR )  {^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^n.addChildToBack ( tmpewBody ) ;^127^^^^^112^142^n.addChildToBack ( newBody ) ;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^n.addChildToBack ( parent ) ;^127^^^^^112^142^n.addChildToBack ( newBody ) ;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Argument_Swapping]^n.addChildToBack ( newBodyewBody ) ;^127^^^^^112^142^n.addChildToBack ( newBody ) ;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Argument_Swapping]^n.addChildToBack ( n ) ;^127^^^^^112^142^n.addChildToBack ( newBody ) ;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Wrong_Operator]^if  ( key.getType (  )  != Token.VAR )  {^70^^^^^55^85^if  ( key.getType (  )  == Token.VAR )  {^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^new Node ( Token.ASSIGN, parent, tmp.cloneTree (  )  )  ) ,^103^104^105^106^^88^118^new Node ( Token.ASSIGN, key, tmp.cloneTree (  )  )  ) ,^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^ifBody = new Node ( Token.BLOCK, parent, new Node (^80^81^82^83^^65^95^ifBody = new Node ( Token.BLOCK, key, new Node (^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^n.addChildToFront ( new Node ( Token.VAR, parent )  ) ;^65^^^^^50^80^n.addChildToFront ( new Node ( Token.VAR, tmp )  ) ;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^new Node ( Token.ASSIGN, tmp.getFirstChild (  ) .cloneNode (  ) , tmp.cloneTree (  )  )  ) ,^85^86^87^88^^70^100^new Node ( Token.ASSIGN, key.getFirstChild (  ) .cloneNode (  ) , tmp.cloneTree (  )  )  ) ,^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^new Node ( Token.ASSIGN, tmp, tmp.cloneTree (  )  )  ) ,^103^104^105^106^^88^118^new Node ( Token.ASSIGN, key, tmp.cloneTree (  )  )  ) ,^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^new Node ( Token.GETPROP, newBody.cloneTree (  ) , Node.newString ( "match" )  ) ,^119^120^121^122^^104^134^new Node ( Token.GETPROP, tmp.cloneTree (  ) , Node.newString ( "match" )  ) ,^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^if  ( newBody.getType (  )  == Token.FOR && n.getChildCount (  )  == 3 )  {^58^^^^^43^73^if  ( n.getType (  )  == Token.FOR && n.getChildCount (  )  == 3 )  {^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^Node body = newBody.getLastChild (  ) ;^59^^^^^44^74^Node body = n.getLastChild (  ) ;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^n.addChildToFront ( newBodyew Node ( Token.VAR, tmp )  ) ;^65^^^^^50^80^n.addChildToFront ( new Node ( Token.VAR, tmp )  ) ;^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^if  ( parent.getType (  )  == Token.VAR )  {^70^^^^^55^85^if  ( key.getType (  )  == Token.VAR )  {^[CLASS] IgnoreCajaProperties Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] AbstractCompiler  compiler  boolean  NodeTraversal  t  int  counter  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^if  ( tmp.getType (  )  == Token.FOR && n.getChildCount (  )  == 3 )  {^58^^^^^43^73^if  ( n.getType (  )  == Token.FOR && n.getChildCount (  )  == 3 )  {^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Wrong_Operator]^if  ( n.getType (  )  == Token.FOR || n.getChildCount (  )  == 3 )  {^58^^^^^43^73^if  ( n.getType (  )  == Token.FOR && n.getChildCount (  )  == 3 )  {^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Wrong_Operator]^if  ( n.getType (  )  != Token.FOR && n.getChildCount (  )  == 3 )  {^58^^^^^43^73^if  ( n.getType (  )  == Token.FOR && n.getChildCount (  )  == 3 )  {^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Wrong_Operator]^if  ( n.getType (  )  == Token.FOR && n.getChildCount (  )  >= 3 )  {^58^^^^^43^73^if  ( n.getType (  )  == Token.FOR && n.getChildCount (  )  == 3 )  {^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Wrong_Operator]^if  ( n.getType (  )  >= Token.FOR && n.getChildCount (  )  == 3 )  {^58^^^^^43^73^if  ( n.getType (  )  == Token.FOR && n.getChildCount (  )  == 3 )  {^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^if  ( newBody.getType (  )  == Token.VAR )  {^70^^^^^55^85^if  ( key.getType (  )  == Token.VAR )  {^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Wrong_Operator]^if  ( key.getType (  )  != Token.VAR )  {^70^^^^^55^85^if  ( key.getType (  )  == Token.VAR )  {^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^new Node ( Token.ASSIGN, parent, tmp.cloneTree (  )  )  ) ,^103^104^105^106^^88^118^new Node ( Token.ASSIGN, key, tmp.cloneTree (  )  )  ) ,^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^new Node ( Token.ASSIGN, key, parent.cloneTree (  )  )  ) ,^103^104^105^106^^88^118^new Node ( Token.ASSIGN, key, tmp.cloneTree (  )  )  ) ,^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Argument_Swapping]^new Node ( Token.ASSIGN, tmp, key.cloneTree (  )  )  ) ,^103^104^105^106^^88^118^new Node ( Token.ASSIGN, key, tmp.cloneTree (  )  )  ) ,^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^ifBody = new Node ( Token.BLOCK, tmp, new Node (^80^81^82^83^^65^95^ifBody = new Node ( Token.BLOCK, key, new Node (^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^ifBody = new Node ( Token.BLOCK, parent, new Node (^80^81^82^83^^65^95^ifBody = new Node ( Token.BLOCK, key, new Node (^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^new Node ( Token.ASSIGN, parent.getFirstChild (  ) .cloneNode (  ) , tmp.cloneTree (  )  )  ) ,^85^86^87^88^^70^100^new Node ( Token.ASSIGN, key.getFirstChild (  ) .cloneNode (  ) , tmp.cloneTree (  )  )  ) ,^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^new Node ( Token.ASSIGN, key.getFirstChild (  ) .cloneNode (  ) , parent.cloneTree (  )  )  ) ,^85^86^87^88^^70^100^new Node ( Token.ASSIGN, key.getFirstChild (  ) .cloneNode (  ) , tmp.cloneTree (  )  )  ) ,^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Argument_Swapping]^new Node ( Token.ASSIGN, tmp.getFirstChild (  ) .cloneNode (  ) , key.cloneTree (  )  )  ) ,^85^86^87^88^^70^100^new Node ( Token.ASSIGN, key.getFirstChild (  ) .cloneNode (  ) , tmp.cloneTree (  )  )  ) ,^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^n.addChildToFront ( new Node ( Token.VAR, parent )  ) ;^65^^^^^50^80^n.addChildToFront ( new Node ( Token.VAR, tmp )  ) ;^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^ifBody = new Node ( Token.BLOCK, newBody, new Node (^80^81^82^83^^65^95^ifBody = new Node ( Token.BLOCK, key, new Node (^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^new Node ( Token.GETPROP, newBody.cloneTree (  ) , Node.newString ( "match" )  ) ,^119^120^121^122^^104^134^new Node ( Token.GETPROP, tmp.cloneTree (  ) , Node.newString ( "match" )  ) ,^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^Node body = tmp.getLastChild (  ) ;^59^^^^^44^74^Node body = n.getLastChild (  ) ;^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^n.removeChild ( parent ) ;^60^^^^^45^75^n.removeChild ( body ) ;^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^Node key = tmp.getFirstChild (  ) ;^61^^^^^46^76^Node key = n.getFirstChild (  ) ;^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^n.removeChild ( parent ) ;^62^^^^^47^77^n.removeChild ( key ) ;^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^n.addChildToFront ( new Node ( Token.VAR, newBody )  ) ;^65^^^^^50^80^n.addChildToFront ( new Node ( Token.VAR, tmp )  ) ;^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Argument_Swapping]^n.addChildToFront ( tmpew Node ( Token.VAR, n )  ) ;^65^^^^^50^80^n.addChildToFront ( new Node ( Token.VAR, tmp )  ) ;^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^if  ( parent.getType (  )  == Token.VAR )  {^70^^^^^55^85^if  ( key.getType (  )  == Token.VAR )  {^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^n.addChildToBack ( tmpewBody ) ;^127^^^^^112^142^n.addChildToBack ( newBody ) ;^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^n.addChildToBack ( parent ) ;^127^^^^^112^142^n.addChildToBack ( newBody ) ;^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Argument_Swapping]^n.addChildToBack ( newBodyewBody ) ;^127^^^^^112^142^n.addChildToBack ( newBody ) ;^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Argument_Swapping]^n.addChildToBack ( n ) ;^127^^^^^112^142^n.addChildToBack ( newBody ) ;^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^if  ( tmp.getType (  )  == Token.VAR )  {^70^^^^^55^85^if  ( key.getType (  )  == Token.VAR )  {^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Wrong_Operator]^if  ( key.getType (  )  >= Token.VAR )  {^70^^^^^55^85^if  ( key.getType (  )  == Token.VAR )  {^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^new Node ( Token.ASSIGN, newBody, tmp.cloneTree (  )  )  ) ,^103^104^105^106^^88^118^new Node ( Token.ASSIGN, key, tmp.cloneTree (  )  )  ) ,^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^new Node ( Token.ASSIGN, tmp.getFirstChild (  ) .cloneNode (  ) , tmp.cloneTree (  )  )  ) ,^85^86^87^88^^70^100^new Node ( Token.ASSIGN, key.getFirstChild (  ) .cloneNode (  ) , tmp.cloneTree (  )  )  ) ,^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^new Node ( Token.ASSIGN, newBody.getFirstChild (  ) .cloneNode (  ) , tmp.cloneTree (  )  )  ) ,^85^86^87^88^^70^100^new Node ( Token.ASSIGN, key.getFirstChild (  ) .cloneNode (  ) , tmp.cloneTree (  )  )  ) ,^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^new Node ( Token.ASSIGN, key, newBody.cloneTree (  )  )  ) ,^103^104^105^106^^88^118^new Node ( Token.ASSIGN, key, tmp.cloneTree (  )  )  ) ,^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^Node body = parent.getLastChild (  ) ;^59^^^^^44^74^Node body = n.getLastChild (  ) ;^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^n.removeChild ( tmp ) ;^60^^^^^45^75^n.removeChild ( body ) ;^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^Node key = newBody.getFirstChild (  ) ;^61^^^^^46^76^Node key = n.getFirstChild (  ) ;^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^n.removeChild ( tmp ) ;^62^^^^^47^77^n.removeChild ( key ) ;^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
[BugLab_Variable_Misuse]^n.addChildToFront ( parentew Node ( Token.VAR, tmp )  ) ;^65^^^^^50^80^n.addChildToFront ( new Node ( Token.VAR, tmp )  ) ;^[CLASS] Traversal  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] boolean  NodeTraversal  t  Node  assignment  body  ifBody  key  n  newBody  parent  tmp  
