[P8_Replace_Mix]^private  final Logger logger = Logger.getLogger ( RemoveUnusedPrototypeProperties.class.getName (  )  ) ;^34^35^^^^34^35^private static final Logger logger = Logger.getLogger ( RemoveUnusedPrototypeProperties.class.getName (  )  ) ;^[CLASS] RemoveUnusedPrototypeProperties   [VARIABLES] 
[P8_Replace_Mix]^private  boolean canModifyExterns;^38^^^^^33^43^private final boolean canModifyExterns;^[CLASS] RemoveUnusedPrototypeProperties   [VARIABLES] 
[P8_Replace_Mix]^private  boolean anchorUnusedVars;^39^^^^^34^44^private final boolean anchorUnusedVars;^[CLASS] RemoveUnusedPrototypeProperties   [VARIABLES] 
[P8_Replace_Mix]^this.compiler =  null;^53^^^^^51^56^this.compiler = compiler;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] <init> [RETURN_TYPE] AbstractCompiler,boolean,boolean)   AbstractCompiler compiler boolean canModifyExterns boolean anchorUnusedVars [VARIABLES] AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  Logger  logger  
[P5_Replace_Variable]^this.canModifyExterns = anchorUnusedVars;^54^^^^^51^56^this.canModifyExterns = canModifyExterns;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] <init> [RETURN_TYPE] AbstractCompiler,boolean,boolean)   AbstractCompiler compiler boolean canModifyExterns boolean anchorUnusedVars [VARIABLES] AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  Logger  logger  
[P5_Replace_Variable]^this.anchorUnusedVars = canModifyExterns;^55^^^^^51^56^this.anchorUnusedVars = anchorUnusedVars;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] <init> [RETURN_TYPE] AbstractCompiler,boolean,boolean)   AbstractCompiler compiler boolean canModifyExterns boolean anchorUnusedVars [VARIABLES] AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  Logger  logger  
[P5_Replace_Variable]^new AnalyzePrototypeProperties (  n analyzer.process ( externRoot, root ) ;^60^61^62^^^58^64^new AnalyzePrototypeProperties ( compiler, n analyzer.process ( externRoot, root ) ;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] process [RETURN_TYPE] void   Node externRoot Node root [VARIABLES] AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  Logger  logger  Node  externRoot  root  AnalyzePrototypeProperties  analyzer  
[P5_Replace_Variable]^AnalyzePrototypeProperties analyzer = new AnalyzePrototypeProperties (  n analyzer.process ( externRoot, root ) ;^59^60^61^62^^58^64^AnalyzePrototypeProperties analyzer = new AnalyzePrototypeProperties ( compiler, n analyzer.process ( externRoot, root ) ;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] process [RETURN_TYPE] void   Node externRoot Node root [VARIABLES] AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  Logger  logger  Node  externRoot  root  AnalyzePrototypeProperties  analyzer  
[P5_Replace_Variable]^analyzer.process (  root ) ;^62^^^^^58^64^analyzer.process ( externRoot, root ) ;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] process [RETURN_TYPE] void   Node externRoot Node root [VARIABLES] AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  Logger  logger  Node  externRoot  root  AnalyzePrototypeProperties  analyzer  
[P5_Replace_Variable]^analyzer.process ( externRoot ) ;^62^^^^^58^64^analyzer.process ( externRoot, root ) ;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] process [RETURN_TYPE] void   Node externRoot Node root [VARIABLES] AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  Logger  logger  Node  externRoot  root  AnalyzePrototypeProperties  analyzer  
[P5_Replace_Variable]^analyzer.process ( root, externRoot ) ;^62^^^^^58^64^analyzer.process ( externRoot, root ) ;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] process [RETURN_TYPE] void   Node externRoot Node root [VARIABLES] AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  Logger  logger  Node  externRoot  root  AnalyzePrototypeProperties  analyzer  
[P14_Delete_Statement]^^62^^^^^58^64^analyzer.process ( externRoot, root ) ;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] process [RETURN_TYPE] void   Node externRoot Node root [VARIABLES] AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  Logger  logger  Node  externRoot  root  AnalyzePrototypeProperties  analyzer  
[P7_Replace_Invocation]^removeUnusedSymbols ( analyzer .process ( root , externRoot )   ) ;^63^^^^^58^64^removeUnusedSymbols ( analyzer.getAllNameInfo (  )  ) ;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] process [RETURN_TYPE] void   Node externRoot Node root [VARIABLES] AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  Logger  logger  Node  externRoot  root  AnalyzePrototypeProperties  analyzer  
[P14_Delete_Statement]^^63^^^^^58^64^removeUnusedSymbols ( analyzer.getAllNameInfo (  )  ) ;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] process [RETURN_TYPE] void   Node externRoot Node root [VARIABLES] AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  Logger  logger  Node  externRoot  root  AnalyzePrototypeProperties  analyzer  
[P3_Replace_Literal]^boolean changed = true;^71^^^^^70^86^boolean changed = false;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] removeUnusedSymbols [RETURN_TYPE] void   NameInfo> allNameInfo [VARIABLES] Collection  allNameInfo  AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  changed  NameInfo  nameInfo  Logger  logger  Symbol  declaration  
[P8_Replace_Mix]^if  ( !nameInfo .getDeclarations (  )   )  {^73^^^^^70^86^if  ( !nameInfo.isReferenced (  )  )  {^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] removeUnusedSymbols [RETURN_TYPE] void   NameInfo> allNameInfo [VARIABLES] Collection  allNameInfo  AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  changed  NameInfo  nameInfo  Logger  logger  Symbol  declaration  
[P15_Unwrap_Block]^for (com.google.javascript.jscomp.AnalyzePrototypeProperties.Symbol declaration : nameInfo.getDeclarations()) {    declaration.remove();    changed = true;}; com.google.javascript.jscomp.RemoveUnusedPrototypeProperties.logger.fine(("Removed unused prototype property: " + (nameInfo.name)));^73^74^75^76^77^70^86^if  ( !nameInfo.isReferenced (  )  )  { for  ( Symbol declaration : nameInfo.getDeclarations (  )  )  { declaration.remove (  ) ; changed = true; }^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] removeUnusedSymbols [RETURN_TYPE] void   NameInfo> allNameInfo [VARIABLES] Collection  allNameInfo  AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  changed  NameInfo  nameInfo  Logger  logger  Symbol  declaration  
[P16_Remove_Block]^^73^74^75^76^77^70^86^if  ( !nameInfo.isReferenced (  )  )  { for  ( Symbol declaration : nameInfo.getDeclarations (  )  )  { declaration.remove (  ) ; changed = true; }^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] removeUnusedSymbols [RETURN_TYPE] void   NameInfo> allNameInfo [VARIABLES] Collection  allNameInfo  AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  changed  NameInfo  nameInfo  Logger  logger  Symbol  declaration  
[P3_Replace_Literal]^changed = false;^76^^^^^70^86^changed = true;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] removeUnusedSymbols [RETURN_TYPE] void   NameInfo> allNameInfo [VARIABLES] Collection  allNameInfo  AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  changed  NameInfo  nameInfo  Logger  logger  Symbol  declaration  
[P14_Delete_Statement]^^74^75^76^77^^70^86^for  ( Symbol declaration : nameInfo.getDeclarations (  )  )  { declaration.remove (  ) ; changed = true; }^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] removeUnusedSymbols [RETURN_TYPE] void   NameInfo> allNameInfo [VARIABLES] Collection  allNameInfo  AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  changed  NameInfo  nameInfo  Logger  logger  Symbol  declaration  
[P14_Delete_Statement]^^75^76^^^^70^86^declaration.remove (  ) ; changed = true;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] removeUnusedSymbols [RETURN_TYPE] void   NameInfo> allNameInfo [VARIABLES] Collection  allNameInfo  AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  changed  NameInfo  nameInfo  Logger  logger  Symbol  declaration  
[P2_Replace_Operator]^logger.fine ( "Removed unused prototype property: "  <<  nameInfo.name ) ;^79^^^^^70^86^logger.fine ( "Removed unused prototype property: " + nameInfo.name ) ;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] removeUnusedSymbols [RETURN_TYPE] void   NameInfo> allNameInfo [VARIABLES] Collection  allNameInfo  AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  changed  NameInfo  nameInfo  Logger  logger  Symbol  declaration  
[P3_Replace_Literal]^logger.fine ( "ed unus" + nameInfo.name ) ;^79^^^^^70^86^logger.fine ( "Removed unused prototype property: " + nameInfo.name ) ;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] removeUnusedSymbols [RETURN_TYPE] void   NameInfo> allNameInfo [VARIABLES] Collection  allNameInfo  AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  changed  NameInfo  nameInfo  Logger  logger  Symbol  declaration  
[P14_Delete_Statement]^^79^^^^^70^86^logger.fine ( "Removed unused prototype property: " + nameInfo.name ) ;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] removeUnusedSymbols [RETURN_TYPE] void   NameInfo> allNameInfo [VARIABLES] Collection  allNameInfo  AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  changed  NameInfo  nameInfo  Logger  logger  Symbol  declaration  
[P8_Replace_Mix]^for  ( Symbol declaration : nameInfo .isReferenced (  )   )  {^74^^^^^70^86^for  ( Symbol declaration : nameInfo.getDeclarations (  )  )  {^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] removeUnusedSymbols [RETURN_TYPE] void   NameInfo> allNameInfo [VARIABLES] Collection  allNameInfo  AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  changed  NameInfo  nameInfo  Logger  logger  Symbol  declaration  
[P2_Replace_Operator]^logger.fine ( "Removed unused prototype property: "  >>  nameInfo.name ) ;^79^^^^^70^86^logger.fine ( "Removed unused prototype property: " + nameInfo.name ) ;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] removeUnusedSymbols [RETURN_TYPE] void   NameInfo> allNameInfo [VARIABLES] Collection  allNameInfo  AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  changed  NameInfo  nameInfo  Logger  logger  Symbol  declaration  
[P3_Replace_Literal]^logger.fine ( "Removed unused prototype property: ype p" + nameInfo.name ) ;^79^^^^^70^86^logger.fine ( "Removed unused prototype property: " + nameInfo.name ) ;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] removeUnusedSymbols [RETURN_TYPE] void   NameInfo> allNameInfo [VARIABLES] Collection  allNameInfo  AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  changed  NameInfo  nameInfo  Logger  logger  Symbol  declaration  
[P5_Replace_Variable]^if  ( canModifyExterns )  {^83^^^^^70^86^if  ( changed )  {^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] removeUnusedSymbols [RETURN_TYPE] void   NameInfo> allNameInfo [VARIABLES] Collection  allNameInfo  AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  changed  NameInfo  nameInfo  Logger  logger  Symbol  declaration  
[P15_Unwrap_Block]^compiler.reportCodeChange();^83^84^85^^^70^86^if  ( changed )  { compiler.reportCodeChange (  ) ; }^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] removeUnusedSymbols [RETURN_TYPE] void   NameInfo> allNameInfo [VARIABLES] Collection  allNameInfo  AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  changed  NameInfo  nameInfo  Logger  logger  Symbol  declaration  
[P16_Remove_Block]^^83^84^85^^^70^86^if  ( changed )  { compiler.reportCodeChange (  ) ; }^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] removeUnusedSymbols [RETURN_TYPE] void   NameInfo> allNameInfo [VARIABLES] Collection  allNameInfo  AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  changed  NameInfo  nameInfo  Logger  logger  Symbol  declaration  
[P14_Delete_Statement]^^84^^^^^70^86^compiler.reportCodeChange (  ) ;^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] removeUnusedSymbols [RETURN_TYPE] void   NameInfo> allNameInfo [VARIABLES] Collection  allNameInfo  AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  changed  NameInfo  nameInfo  Logger  logger  Symbol  declaration  
[P13_Insert_Block]^if  ( changed )  {     compiler.reportCodeChange (  ) ; }^84^^^^^70^86^[Delete]^[CLASS] RemoveUnusedPrototypeProperties  [METHOD] removeUnusedSymbols [RETURN_TYPE] void   NameInfo> allNameInfo [VARIABLES] Collection  allNameInfo  AbstractCompiler  compiler  boolean  anchorUnusedVars  canModifyExterns  changed  NameInfo  nameInfo  Logger  logger  Symbol  declaration  