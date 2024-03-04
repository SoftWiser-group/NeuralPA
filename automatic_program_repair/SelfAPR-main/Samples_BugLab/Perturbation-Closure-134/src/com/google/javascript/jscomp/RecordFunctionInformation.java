[BugLab_Argument_Swapping]^NodeTraversal.traverse ( root, compiler, this ) ;^63^^^^^62^78^NodeTraversal.traverse ( compiler, root, this ) ;^[CLASS] RecordFunctionInformation 1  [METHOD] process [RETURN_TYPE] void   Node externs Node root [VARIABLES] Compiler  compiler  boolean  Builder  mapBuilder  Node  externs  root  JSModule  m  o1  o2  JSModuleGraph  graph  FunctionNames  functionNames  
[BugLab_Wrong_Operator]^if  ( graph != null )  {^66^^^^^62^78^if  ( graph == null )  {^[CLASS] RecordFunctionInformation 1  [METHOD] process [RETURN_TYPE] void   Node externs Node root [VARIABLES] Compiler  compiler  boolean  Builder  mapBuilder  Node  externs  root  JSModule  m  o1  o2  JSModuleGraph  graph  FunctionNames  functionNames  
[BugLab_Variable_Misuse]^return m.getName (  ) .compareTo ( o2.getName (  )  ) ;^72^^^^^62^78^return o1.getName (  ) .compareTo ( o2.getName (  )  ) ;^[CLASS] RecordFunctionInformation 1  [METHOD] process [RETURN_TYPE] void   Node externs Node root [VARIABLES] Compiler  compiler  boolean  Builder  mapBuilder  Node  externs  root  JSModule  m  o1  o2  JSModuleGraph  graph  FunctionNames  functionNames  
[BugLab_Variable_Misuse]^return o1.getName (  ) .compareTo ( m.getName (  )  ) ;^72^^^^^62^78^return o1.getName (  ) .compareTo ( o2.getName (  )  ) ;^[CLASS] RecordFunctionInformation 1  [METHOD] process [RETURN_TYPE] void   Node externs Node root [VARIABLES] Compiler  compiler  boolean  Builder  mapBuilder  Node  externs  root  JSModule  m  o1  o2  JSModuleGraph  graph  FunctionNames  functionNames  
[BugLab_Argument_Swapping]^return o2.getName (  ) .compareTo ( o1.getName (  )  ) ;^72^^^^^62^78^return o1.getName (  ) .compareTo ( o2.getName (  )  ) ;^[CLASS] RecordFunctionInformation 1  [METHOD] process [RETURN_TYPE] void   Node externs Node root [VARIABLES] Compiler  compiler  boolean  Builder  mapBuilder  Node  externs  root  JSModule  m  o1  o2  JSModuleGraph  graph  FunctionNames  functionNames  
[BugLab_Variable_Misuse]^return o2.getName (  ) .compareTo ( o2.getName (  )  ) ;^72^^^^^62^78^return o1.getName (  ) .compareTo ( o2.getName (  )  ) ;^[CLASS] RecordFunctionInformation 1  [METHOD] process [RETURN_TYPE] void   Node externs Node root [VARIABLES] Compiler  compiler  boolean  Builder  mapBuilder  Node  externs  root  JSModule  m  o1  o2  JSModuleGraph  graph  FunctionNames  functionNames  
[BugLab_Variable_Misuse]^return o1.getName (  ) .compareTo ( o1.getName (  )  ) ;^72^^^^^62^78^return o1.getName (  ) .compareTo ( o2.getName (  )  ) ;^[CLASS] RecordFunctionInformation 1  [METHOD] process [RETURN_TYPE] void   Node externs Node root [VARIABLES] Compiler  compiler  boolean  Builder  mapBuilder  Node  externs  root  JSModule  m  o1  o2  JSModuleGraph  graph  FunctionNames  functionNames  
[BugLab_Variable_Misuse]^addModuleInformation ( o2 ) ;^75^^^^^62^78^addModuleInformation ( m ) ;^[CLASS] RecordFunctionInformation 1  [METHOD] process [RETURN_TYPE] void   Node externs Node root [VARIABLES] Compiler  compiler  boolean  Builder  mapBuilder  Node  externs  root  JSModule  m  o1  o2  JSModuleGraph  graph  FunctionNames  functionNames  
[BugLab_Argument_Swapping]^return o2.getName (  ) .compareTo ( o1.getName (  )  ) ;^72^^^^^71^73^return o1.getName (  ) .compareTo ( o2.getName (  )  ) ;^[CLASS] RecordFunctionInformation 1  [METHOD] compare [RETURN_TYPE] int   JSModule o1 JSModule o2 [VARIABLES] Compiler  compiler  JSModule  o1  o2  boolean  Builder  mapBuilder  FunctionNames  functionNames  
[BugLab_Variable_Misuse]^return o2.getName (  ) .compareTo ( o2.getName (  )  ) ;^72^^^^^71^73^return o1.getName (  ) .compareTo ( o2.getName (  )  ) ;^[CLASS] RecordFunctionInformation 1  [METHOD] compare [RETURN_TYPE] int   JSModule o1 JSModule o2 [VARIABLES] Compiler  compiler  JSModule  o1  o2  boolean  Builder  mapBuilder  FunctionNames  functionNames  
[BugLab_Variable_Misuse]^return o1.getName (  ) .compareTo ( o1.getName (  )  ) ;^72^^^^^71^73^return o1.getName (  ) .compareTo ( o2.getName (  )  ) ;^[CLASS] RecordFunctionInformation 1  [METHOD] compare [RETURN_TYPE] int   JSModule o1 JSModule o2 [VARIABLES] Compiler  compiler  JSModule  o1  o2  boolean  Builder  mapBuilder  FunctionNames  functionNames  
[BugLab_Variable_Misuse]^if  ( parent.getType (  )  != Token.FUNCTION )  {^82^^^^^81^102^if  ( n.getType (  )  != Token.FUNCTION )  {^[CLASS] RecordFunctionInformation 1  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] Compiler  compiler  boolean  NodeTraversal  t  Builder  mapBuilder  Node  n  parent  JSModule  module  String  compiledSource  int  id  FunctionNames  functionNames  
[BugLab_Wrong_Operator]^if  ( n.getType (  )  < Token.FUNCTION )  {^82^^^^^81^102^if  ( n.getType (  )  != Token.FUNCTION )  {^[CLASS] RecordFunctionInformation 1  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] Compiler  compiler  boolean  NodeTraversal  t  Builder  mapBuilder  Node  n  parent  JSModule  module  String  compiledSource  int  id  FunctionNames  functionNames  
[BugLab_Variable_Misuse]^int id = functionNames.getFunctionId ( parent ) ;^86^^^^^81^102^int id = functionNames.getFunctionId ( n ) ;^[CLASS] RecordFunctionInformation 1  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] Compiler  compiler  boolean  NodeTraversal  t  Builder  mapBuilder  Node  n  parent  JSModule  module  String  compiledSource  int  id  FunctionNames  functionNames  
[BugLab_Argument_Swapping]^int id = n.getFunctionId ( functionNames ) ;^86^^^^^81^102^int id = functionNames.getFunctionId ( n ) ;^[CLASS] RecordFunctionInformation 1  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] Compiler  compiler  boolean  NodeTraversal  t  Builder  mapBuilder  Node  n  parent  JSModule  module  String  compiledSource  int  id  FunctionNames  functionNames  
[BugLab_Wrong_Operator]^if  ( id <= 0 )  {^87^^^^^81^102^if  ( id < 0 )  {^[CLASS] RecordFunctionInformation 1  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] Compiler  compiler  boolean  NodeTraversal  t  Builder  mapBuilder  Node  n  parent  JSModule  module  String  compiledSource  int  id  FunctionNames  functionNames  
[BugLab_Wrong_Literal]^if  ( id < id )  {^87^^^^^81^102^if  ( id < 0 )  {^[CLASS] RecordFunctionInformation 1  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] Compiler  compiler  boolean  NodeTraversal  t  Builder  mapBuilder  Node  n  parent  JSModule  module  String  compiledSource  int  id  FunctionNames  functionNames  
[BugLab_Variable_Misuse]^String compiledSource = compiler.toSource ( parent ) ;^92^^^^^81^102^String compiledSource = compiler.toSource ( n ) ;^[CLASS] RecordFunctionInformation 1  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] Compiler  compiler  boolean  NodeTraversal  t  Builder  mapBuilder  Node  n  parent  JSModule  module  String  compiledSource  int  id  FunctionNames  functionNames  
[BugLab_Argument_Swapping]^String compiledSource = n.toSource ( compiler ) ;^92^^^^^81^102^String compiledSource = compiler.toSource ( n ) ;^[CLASS] RecordFunctionInformation 1  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] Compiler  compiler  boolean  NodeTraversal  t  Builder  mapBuilder  Node  n  parent  JSModule  module  String  compiledSource  int  id  FunctionNames  functionNames  
[BugLab_Variable_Misuse]^mapBuilder.addEntry ( FunctionInformationMap.Entry.newBuilder (  ) .setId ( id ) .setSourceName ( t.getSourceName (  )  ) .setLineNumber ( parent.getLineno (  )  )^94^95^96^97^^81^102^mapBuilder.addEntry ( FunctionInformationMap.Entry.newBuilder (  ) .setId ( id ) .setSourceName ( t.getSourceName (  )  ) .setLineNumber ( n.getLineno (  )  )^[CLASS] RecordFunctionInformation 1  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] Compiler  compiler  boolean  NodeTraversal  t  Builder  mapBuilder  Node  n  parent  JSModule  module  String  compiledSource  int  id  FunctionNames  functionNames  
[BugLab_Argument_Swapping]^mapBuilder.addEntry ( FunctionInformationMap.Entry.newBuilder (  ) .setId ( n ) .setSourceName ( t.getSourceName (  )  ) .setLineNumber ( id.getLineno (  )  )^94^95^96^97^^81^102^mapBuilder.addEntry ( FunctionInformationMap.Entry.newBuilder (  ) .setId ( id ) .setSourceName ( t.getSourceName (  )  ) .setLineNumber ( n.getLineno (  )  )^[CLASS] RecordFunctionInformation 1  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] Compiler  compiler  boolean  NodeTraversal  t  Builder  mapBuilder  Node  n  parent  JSModule  module  String  compiledSource  int  id  FunctionNames  functionNames  
[BugLab_Argument_Swapping]^mapBuilder.addEntry ( FunctionInformationMap.Entry.newBuilder (  ) .setId ( t ) .setSourceName ( id.getSourceName (  )  ) .setLineNumber ( n.getLineno (  )  )^94^95^96^97^^81^102^mapBuilder.addEntry ( FunctionInformationMap.Entry.newBuilder (  ) .setId ( id ) .setSourceName ( t.getSourceName (  )  ) .setLineNumber ( n.getLineno (  )  )^[CLASS] RecordFunctionInformation 1  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] Compiler  compiler  boolean  NodeTraversal  t  Builder  mapBuilder  Node  n  parent  JSModule  module  String  compiledSource  int  id  FunctionNames  functionNames  
[BugLab_Argument_Swapping]^mapBuilder.addEntry ( FunctionInformationMap.Entry.newBuilder (  ) .setId ( id ) .setSourceName ( n.getSourceName (  )  ) .setLineNumber ( t.getLineno (  )  )^94^95^96^97^^81^102^mapBuilder.addEntry ( FunctionInformationMap.Entry.newBuilder (  ) .setId ( id ) .setSourceName ( t.getSourceName (  )  ) .setLineNumber ( n.getLineno (  )  )^[CLASS] RecordFunctionInformation 1  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] Compiler  compiler  boolean  NodeTraversal  t  Builder  mapBuilder  Node  n  parent  JSModule  module  String  compiledSource  int  id  FunctionNames  functionNames  
[BugLab_Variable_Misuse]^.setLineNumber ( parent.getLineno (  )  ) .setModuleName ( module == null ? "" : module.getName (  )  ) .setSize ( compiledSource.length (  )  ) .setName ( functionNames.getFunctionName ( n )  )^97^98^99^100^^81^102^.setLineNumber ( n.getLineno (  )  ) .setModuleName ( module == null ? "" : module.getName (  )  ) .setSize ( compiledSource.length (  )  ) .setName ( functionNames.getFunctionName ( n )  )^[CLASS] RecordFunctionInformation 1  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] Compiler  compiler  boolean  NodeTraversal  t  Builder  mapBuilder  Node  n  parent  JSModule  module  String  compiledSource  int  id  FunctionNames  functionNames  
[BugLab_Variable_Misuse]^.setName ( functionNames.getFunctionName ( parent )  ) .setCompiledSource ( compiledSource ) .build (  )  ) ;^100^101^^^^81^102^.setName ( functionNames.getFunctionName ( n )  ) .setCompiledSource ( compiledSource ) .build (  )  ) ;^[CLASS] RecordFunctionInformation 1  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] Compiler  compiler  boolean  NodeTraversal  t  Builder  mapBuilder  Node  n  parent  JSModule  module  String  compiledSource  int  id  FunctionNames  functionNames  
[BugLab_Argument_Swapping]^.setName ( n.getFunctionName ( functionNames )  ) .setCompiledSource ( compiledSource ) .build (  )  ) ;^100^101^^^^81^102^.setName ( functionNames.getFunctionName ( n )  ) .setCompiledSource ( compiledSource ) .build (  )  ) ;^[CLASS] RecordFunctionInformation 1  [METHOD] visit [RETURN_TYPE] void   NodeTraversal t Node n Node parent [VARIABLES] Compiler  compiler  boolean  NodeTraversal  t  Builder  mapBuilder  Node  n  parent  JSModule  module  String  compiledSource  int  id  FunctionNames  functionNames  
[BugLab_Wrong_Operator]^if  ( module == null )  {^118^^^^^115^129^if  ( module != null )  {^[CLASS] RecordFunctionInformation 1  [METHOD] addModuleInformation [RETURN_TYPE] void   JSModule module [VARIABLES] Compiler  compiler  JSModule  module  String  name  source  boolean  Builder  mapBuilder  FunctionNames  functionNames  
[BugLab_Argument_Swapping]^source = module.toSource ( compiler ) ;^120^^^^^115^129^source = compiler.toSource ( module ) ;^[CLASS] RecordFunctionInformation 1  [METHOD] addModuleInformation [RETURN_TYPE] void   JSModule module [VARIABLES] Compiler  compiler  JSModule  module  String  name  source  boolean  Builder  mapBuilder  FunctionNames  functionNames  
[BugLab_Variable_Misuse]^mapBuilder.addModule ( FunctionInformationMap.Module.newBuilder (  ) .setName ( source ) .setCompiledSource ( source ) .build (  )  ) ;^126^127^128^^^115^129^mapBuilder.addModule ( FunctionInformationMap.Module.newBuilder (  ) .setName ( name ) .setCompiledSource ( source ) .build (  )  ) ;^[CLASS] RecordFunctionInformation 1  [METHOD] addModuleInformation [RETURN_TYPE] void   JSModule module [VARIABLES] Compiler  compiler  JSModule  module  String  name  source  boolean  Builder  mapBuilder  FunctionNames  functionNames  
[BugLab_Argument_Swapping]^mapBuilder.addModule ( FunctionInformationMap.Module.newBuilder (  ) .setName ( source ) .setCompiledSource ( name ) .build (  )  ) ;^126^127^128^^^115^129^mapBuilder.addModule ( FunctionInformationMap.Module.newBuilder (  ) .setName ( name ) .setCompiledSource ( source ) .build (  )  ) ;^[CLASS] RecordFunctionInformation 1  [METHOD] addModuleInformation [RETURN_TYPE] void   JSModule module [VARIABLES] Compiler  compiler  JSModule  module  String  name  source  boolean  Builder  mapBuilder  FunctionNames  functionNames  
[BugLab_Argument_Swapping]^return o2.getName (  ) .compareTo ( o1.getName (  )  ) ;^72^^^^^71^73^return o1.getName (  ) .compareTo ( o2.getName (  )  ) ;^[CLASS] 1  [METHOD] compare [RETURN_TYPE] int   JSModule o1 JSModule o2 [VARIABLES] boolean  JSModule  o1  o2  
[BugLab_Variable_Misuse]^return o2.getName (  ) .compareTo ( o2.getName (  )  ) ;^72^^^^^71^73^return o1.getName (  ) .compareTo ( o2.getName (  )  ) ;^[CLASS] 1  [METHOD] compare [RETURN_TYPE] int   JSModule o1 JSModule o2 [VARIABLES] boolean  JSModule  o1  o2  
[BugLab_Variable_Misuse]^return o1.getName (  ) .compareTo ( o1.getName (  )  ) ;^72^^^^^71^73^return o1.getName (  ) .compareTo ( o2.getName (  )  ) ;^[CLASS] 1  [METHOD] compare [RETURN_TYPE] int   JSModule o1 JSModule o2 [VARIABLES] boolean  JSModule  o1  o2  
