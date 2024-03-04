[BugLab_Variable_Misuse]^return 0;^51^^^^^49^52^return annotationNames;^[CLASS] ParserRunner  [METHOD] getAnnotationNames [RETURN_TYPE] Set   [VARIABLES] Set  annotationNames  String  configResource  boolean  
[BugLab_Wrong_Operator]^if  ( annotationNames == null )  {^55^^^^^54^66^if  ( annotationNames != null )  {^[CLASS] ParserRunner  [METHOD] initAnnotationNames [RETURN_TYPE] void   [VARIABLES] ResourceBundle  config  Set  annotationNames  trimmedNames  String  configResource  name  String[]  names  boolean  
[BugLab_Variable_Misuse]^ResourceBundle config = ResourceBundle.getBundle ( name ) ;^60^^^^^54^66^ResourceBundle config = ResourceBundle.getBundle ( configResource ) ;^[CLASS] ParserRunner  [METHOD] initAnnotationNames [RETURN_TYPE] void   [VARIABLES] ResourceBundle  config  Set  annotationNames  trimmedNames  String  configResource  name  String[]  names  boolean  
[BugLab_Variable_Misuse]^trimmedNames.add ( configResource.trim (  )  ) ;^63^^^^^54^66^trimmedNames.add ( name.trim (  )  ) ;^[CLASS] ParserRunner  [METHOD] initAnnotationNames [RETURN_TYPE] void   [VARIABLES] ResourceBundle  config  Set  annotationNames  trimmedNames  String  configResource  name  String[]  names  boolean  
[BugLab_Wrong_Literal]^compilerEnv.setRecordingComments ( false ) ;^93^^^^^78^108^compilerEnv.setRecordingComments ( true ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Wrong_Literal]^compilerEnv.setRecordingLocalJsDocComments ( false ) ;^94^^^^^79^109^compilerEnv.setRecordingLocalJsDocComments ( true ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Wrong_Literal]^compilerEnv.setWarnTrailingComma ( false ) ;^95^^^^^80^110^compilerEnv.setWarnTrailingComma ( true ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Wrong_Literal]^compilerEnv.setReservedKeywordAsIdentifier ( false ) ;^97^^^^^82^112^compilerEnv.setReservedKeywordAsIdentifier ( true ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Wrong_Literal]^compilerEnv.setAllowMemberExprAsFunctionName ( false ) ;^98^^^^^83^113^compilerEnv.setAllowMemberExprAsFunctionName ( true ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Argument_Swapping]^Parser p = new Parser ( errorReporter, compilerEnv ) ;^101^^^^^86^116^Parser p = new Parser ( compilerEnv, errorReporter ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Variable_Misuse]^astRoot = p.parse ( name, sourceName, 1 ) ;^104^^^^^89^119^astRoot = p.parse ( sourceString, sourceName, 1 ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Variable_Misuse]^astRoot = p.parse ( sourceString, name, 1 ) ;^104^^^^^89^119^astRoot = p.parse ( sourceString, sourceName, 1 ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Argument_Swapping]^astRoot = sourceString.parse ( p, sourceName, 1 ) ;^104^^^^^89^119^astRoot = p.parse ( sourceString, sourceName, 1 ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Argument_Swapping]^astRoot = p.parse ( sourceName, sourceString, 1 ) ;^104^^^^^89^119^astRoot = p.parse ( sourceString, sourceName, 1 ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Wrong_Literal]^astRoot = p.parse ( sourceString, sourceName, 2 ) ;^104^^^^^89^119^astRoot = p.parse ( sourceString, sourceName, 1 ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Variable_Misuse]^astRoot = p.parse ( sourceString, sourceString, 1 ) ;^104^^^^^89^119^astRoot = p.parse ( sourceString, sourceName, 1 ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Argument_Swapping]^astRoot = sourceName.parse ( sourceString, p, 1 ) ;^104^^^^^89^119^astRoot = p.parse ( sourceString, sourceName, 1 ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Variable_Misuse]^astRoot = p.parse ( sourceName, sourceName, 1 ) ;^104^^^^^89^119^astRoot = p.parse ( sourceString, sourceName, 1 ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Wrong_Literal]^astRoot = p.parse ( sourceString, sourceName, 0 ) ;^104^^^^^89^119^astRoot = p.parse ( sourceString, sourceName, 1 ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Variable_Misuse]^logger.info ( "Error parsing " + sourceString + ": " + e.getMessage (  )  ) ;^106^^^^^91^121^logger.info ( "Error parsing " + sourceName + ": " + e.getMessage (  )  ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Wrong_Operator]^logger.info ( "Error parsing " + sourceName + ": " + e.getMessage (  !=  )  ) ;^106^^^^^91^121^logger.info ( "Error parsing " + sourceName + ": " + e.getMessage (  )  ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Wrong_Operator]^logger.info ( "Error parsing "  >=  sourceName  >=  ": " + e.getMessage (  )  ) ;^106^^^^^91^121^logger.info ( "Error parsing " + sourceName + ": " + e.getMessage (  )  ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Wrong_Operator]^logger.info ( "Error parsing "  ||  sourceName + ": " + e.getMessage (  )  ) ;^106^^^^^91^121^logger.info ( "Error parsing " + sourceName + ": " + e.getMessage (  )  ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Wrong_Operator]^if  ( astRoot == null )  {^111^^^^^96^126^if  ( astRoot != null )  {^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Variable_Misuse]^root = IRFactory.transformTree ( astRoot, sourceName, config, errorReporter ) ;^114^115^^^^99^129^root = IRFactory.transformTree ( astRoot, sourceString, config, errorReporter ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Argument_Swapping]^root = IRFactory.transformTree ( errorReporter, sourceString, config, astRoot ) ;^114^115^^^^99^129^root = IRFactory.transformTree ( astRoot, sourceString, config, errorReporter ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Argument_Swapping]^root = IRFactory.transformTree ( astRoot, config, sourceString, errorReporter ) ;^114^115^^^^99^129^root = IRFactory.transformTree ( astRoot, sourceString, config, errorReporter ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Argument_Swapping]^root = IRFactory.transformTree ( astRoot, sourceString, errorReporter, config ) ;^114^115^^^^99^129^root = IRFactory.transformTree ( astRoot, sourceString, config, errorReporter ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Variable_Misuse]^Config config = new Config ( typeRegistry, 3, isIdeMode ) ;^112^113^^^^97^127^Config config = new Config ( typeRegistry, annotationNames, isIdeMode ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Argument_Swapping]^Config config = new Config ( annotationNames, typeRegistry, isIdeMode ) ;^112^113^^^^97^127^Config config = new Config ( typeRegistry, annotationNames, isIdeMode ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Argument_Swapping]^Config config = new Config ( isIdeMode, annotationNames, typeRegistry ) ;^112^113^^^^97^127^Config config = new Config ( typeRegistry, annotationNames, isIdeMode ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Argument_Swapping]^Config config = new Config ( typeRegistry, isIdeMode, annotationNames ) ;^112^113^^^^97^127^Config config = new Config ( typeRegistry, annotationNames, isIdeMode ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Argument_Swapping]^root = IRFactory.transformTree ( sourceString, astRoot, config, errorReporter ) ;^114^115^^^^99^129^root = IRFactory.transformTree ( astRoot, sourceString, config, errorReporter ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Argument_Swapping]^root = IRFactory.transformTree ( config, sourceString, astRoot, errorReporter ) ;^114^115^^^^99^129^root = IRFactory.transformTree ( astRoot, sourceString, config, errorReporter ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Argument_Swapping]^root = IRFactory.transformTree ( astRoot, errorReporter, config, sourceString ) ;^114^115^^^^99^129^root = IRFactory.transformTree ( astRoot, sourceString, config, errorReporter ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Wrong_Literal]^root.setIsSyntheticBlock ( false ) ;^116^^^^^101^131^root.setIsSyntheticBlock ( true ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
[BugLab_Variable_Misuse]^Config config = new Config ( typeRegistry, null, isIdeMode ) ;^112^113^^^^97^127^Config config = new Config ( typeRegistry, annotationNames, isIdeMode ) ;^[CLASS] ParserRunner  [METHOD] parse [RETURN_TYPE] Node   String sourceName String sourceString boolean isIdeMode JSTypeRegistry typeRegistry ErrorReporter errorReporter Logger logger [VARIABLES] Context  cx  Set  annotationNames  trimmedNames  Config  config  Node  root  CompilerEnvirons  compilerEnv  String  configResource  name  sourceName  sourceString  Logger  logger  Parser  p  JSTypeRegistry  typeRegistry  ErrorReporter  errorReporter  boolean  isIdeMode  EvaluatorException  e  AstRoot  astRoot  
