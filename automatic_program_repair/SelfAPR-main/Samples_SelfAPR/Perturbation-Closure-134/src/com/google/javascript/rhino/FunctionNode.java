[P1_Replace_Type]^private static final  short  serialVersionUID = 1L;^44^^^^^39^49^private static final long serialVersionUID = 1L;^[CLASS] FunctionNode   [VARIABLES] 
[P8_Replace_Mix]^private static final long serialVersionUID = 1;^44^^^^^39^49^private static final long serialVersionUID = 1L;^[CLASS] FunctionNode   [VARIABLES] 
[P1_Replace_Type]^public static final  long  FUNCTION_STATEMENT            = 1;^83^^^^^78^88^public static final int FUNCTION_STATEMENT            = 1;^[CLASS] FunctionNode   [VARIABLES] 
[P3_Replace_Literal]^public static final int FUNCTION_STATEMENT            = -2;^83^^^^^78^88^public static final int FUNCTION_STATEMENT            = 1;^[CLASS] FunctionNode   [VARIABLES] 
[P8_Replace_Mix]^public static final int FUNCTION_STATEMENT            = 1 << 0;^83^^^^^78^88^public static final int FUNCTION_STATEMENT            = 1;^[CLASS] FunctionNode   [VARIABLES] 
[P1_Replace_Type]^public static final  long  FUNCTION_EXPRESSION           = 2;^84^^^^^79^89^public static final int FUNCTION_EXPRESSION           = 2;^[CLASS] FunctionNode   [VARIABLES] 
[P3_Replace_Literal]^public static final int FUNCTION_EXPRESSION           = 0;^84^^^^^79^89^public static final int FUNCTION_EXPRESSION           = 2;^[CLASS] FunctionNode   [VARIABLES] 
[P8_Replace_Mix]^public static final int FUNCTION_EXPRESSION           = 2 * 1;^84^^^^^79^89^public static final int FUNCTION_EXPRESSION           = 2;^[CLASS] FunctionNode   [VARIABLES] 
[P1_Replace_Type]^public static final  long  FUNCTION_EXPRESSION_STATEMENT = 3;^85^^^^^80^90^public static final int FUNCTION_EXPRESSION_STATEMENT = 3;^[CLASS] FunctionNode   [VARIABLES] 
[P3_Replace_Literal]^public static final int FUNCTION_EXPRESSION_STATEMENT = -1;^85^^^^^80^90^public static final int FUNCTION_EXPRESSION_STATEMENT = 3;^[CLASS] FunctionNode   [VARIABLES] 
[P8_Replace_Mix]^public static final  short  FUNCTION_EXPRESSION_STATEMENT = 3 >>> 3;^85^^^^^80^90^public static final int FUNCTION_EXPRESSION_STATEMENT = 3;^[CLASS] FunctionNode   [VARIABLES] 
[P1_Replace_Type]^char functionName;^91^^^^^86^96^String functionName;^[CLASS] FunctionNode   [VARIABLES] 
[P1_Replace_Type]^long  itsFunctionType;^93^^^^^88^98^int itsFunctionType;^[CLASS] FunctionNode   [VARIABLES] 
[P14_Delete_Statement]^^47^^^^^46^49^super ( Token.FUNCTION ) ;^[CLASS] FunctionNode  [METHOD] <init> [RETURN_TYPE] String)   String name [VARIABLES] boolean  itsIgnoreDynamicScope  itsNeedsActivation  String  functionName  name  long  serialVersionUID  int  FUNCTION_EXPRESSION  FUNCTION_EXPRESSION_STATEMENT  FUNCTION_STATEMENT  itsFunctionType  
[P11_Insert_Donor_Statement]^super ( Token.FUNCTION, lineno, charno ) ;super ( Token.FUNCTION ) ;^47^^^^^46^49^super ( Token.FUNCTION ) ;^[CLASS] FunctionNode  [METHOD] <init> [RETURN_TYPE] String)   String name [VARIABLES] boolean  itsIgnoreDynamicScope  itsNeedsActivation  String  functionName  name  long  serialVersionUID  int  FUNCTION_EXPRESSION  FUNCTION_EXPRESSION_STATEMENT  FUNCTION_STATEMENT  itsFunctionType  
[P5_Replace_Variable]^functionName = functionName;^48^^^^^46^49^functionName = name;^[CLASS] FunctionNode  [METHOD] <init> [RETURN_TYPE] String)   String name [VARIABLES] boolean  itsIgnoreDynamicScope  itsNeedsActivation  String  functionName  name  long  serialVersionUID  int  FUNCTION_EXPRESSION  FUNCTION_EXPRESSION_STATEMENT  FUNCTION_STATEMENT  itsFunctionType  
[P8_Replace_Mix]^functionName =  null;^48^^^^^46^49^functionName = name;^[CLASS] FunctionNode  [METHOD] <init> [RETURN_TYPE] String)   String name [VARIABLES] boolean  itsIgnoreDynamicScope  itsNeedsActivation  String  functionName  name  long  serialVersionUID  int  FUNCTION_EXPRESSION  FUNCTION_EXPRESSION_STATEMENT  FUNCTION_STATEMENT  itsFunctionType  
[P5_Replace_Variable]^super ( Token.FUNCTION,  charno ) ;^52^^^^^51^54^super ( Token.FUNCTION, lineno, charno ) ;^[CLASS] FunctionNode  [METHOD] <init> [RETURN_TYPE] String,int,int)   String name int lineno int charno [VARIABLES] boolean  itsIgnoreDynamicScope  itsNeedsActivation  String  functionName  name  long  serialVersionUID  int  FUNCTION_EXPRESSION  FUNCTION_EXPRESSION_STATEMENT  FUNCTION_STATEMENT  charno  itsFunctionType  lineno  
[P5_Replace_Variable]^super ( Token.FUNCTION, lineno ) ;^52^^^^^51^54^super ( Token.FUNCTION, lineno, charno ) ;^[CLASS] FunctionNode  [METHOD] <init> [RETURN_TYPE] String,int,int)   String name int lineno int charno [VARIABLES] boolean  itsIgnoreDynamicScope  itsNeedsActivation  String  functionName  name  long  serialVersionUID  int  FUNCTION_EXPRESSION  FUNCTION_EXPRESSION_STATEMENT  FUNCTION_STATEMENT  charno  itsFunctionType  lineno  
[P5_Replace_Variable]^super ( Token.FUNCTION, charno, lineno ) ;^52^^^^^51^54^super ( Token.FUNCTION, lineno, charno ) ;^[CLASS] FunctionNode  [METHOD] <init> [RETURN_TYPE] String,int,int)   String name int lineno int charno [VARIABLES] boolean  itsIgnoreDynamicScope  itsNeedsActivation  String  functionName  name  long  serialVersionUID  int  FUNCTION_EXPRESSION  FUNCTION_EXPRESSION_STATEMENT  FUNCTION_STATEMENT  charno  itsFunctionType  lineno  
[P8_Replace_Mix]^super ( Token.FUNCTION, itsFunctionType, charno ) ;^52^^^^^51^54^super ( Token.FUNCTION, lineno, charno ) ;^[CLASS] FunctionNode  [METHOD] <init> [RETURN_TYPE] String,int,int)   String name int lineno int charno [VARIABLES] boolean  itsIgnoreDynamicScope  itsNeedsActivation  String  functionName  name  long  serialVersionUID  int  FUNCTION_EXPRESSION  FUNCTION_EXPRESSION_STATEMENT  FUNCTION_STATEMENT  charno  itsFunctionType  lineno  
[P14_Delete_Statement]^^52^53^^^^51^54^super ( Token.FUNCTION, lineno, charno ) ; functionName = name;^[CLASS] FunctionNode  [METHOD] <init> [RETURN_TYPE] String,int,int)   String name int lineno int charno [VARIABLES] boolean  itsIgnoreDynamicScope  itsNeedsActivation  String  functionName  name  long  serialVersionUID  int  FUNCTION_EXPRESSION  FUNCTION_EXPRESSION_STATEMENT  FUNCTION_STATEMENT  charno  itsFunctionType  lineno  
[P11_Insert_Donor_Statement]^super ( Token.FUNCTION ) ;super ( Token.FUNCTION, lineno, charno ) ;^52^^^^^51^54^super ( Token.FUNCTION, lineno, charno ) ;^[CLASS] FunctionNode  [METHOD] <init> [RETURN_TYPE] String,int,int)   String name int lineno int charno [VARIABLES] boolean  itsIgnoreDynamicScope  itsNeedsActivation  String  functionName  name  long  serialVersionUID  int  FUNCTION_EXPRESSION  FUNCTION_EXPRESSION_STATEMENT  FUNCTION_STATEMENT  charno  itsFunctionType  lineno  
[P5_Replace_Variable]^functionName = functionName;^53^^^^^51^54^functionName = name;^[CLASS] FunctionNode  [METHOD] <init> [RETURN_TYPE] String,int,int)   String name int lineno int charno [VARIABLES] boolean  itsIgnoreDynamicScope  itsNeedsActivation  String  functionName  name  long  serialVersionUID  int  FUNCTION_EXPRESSION  FUNCTION_EXPRESSION_STATEMENT  FUNCTION_STATEMENT  charno  itsFunctionType  lineno  
[P8_Replace_Mix]^functionName =  null;^53^^^^^51^54^functionName = name;^[CLASS] FunctionNode  [METHOD] <init> [RETURN_TYPE] String,int,int)   String name int lineno int charno [VARIABLES] boolean  itsIgnoreDynamicScope  itsNeedsActivation  String  functionName  name  long  serialVersionUID  int  FUNCTION_EXPRESSION  FUNCTION_EXPRESSION_STATEMENT  FUNCTION_STATEMENT  charno  itsFunctionType  lineno  
[P5_Replace_Variable]^return name;^57^^^^^56^58^return functionName;^[CLASS] FunctionNode  [METHOD] getFunctionName [RETURN_TYPE] String   [VARIABLES] boolean  itsIgnoreDynamicScope  itsNeedsActivation  String  functionName  name  long  serialVersionUID  int  FUNCTION_EXPRESSION  FUNCTION_EXPRESSION_STATEMENT  FUNCTION_STATEMENT  charno  itsFunctionType  lineno  
[P5_Replace_Variable]^return itsIgnoreDynamicScope;^61^^^^^60^62^return itsNeedsActivation;^[CLASS] FunctionNode  [METHOD] requiresActivation [RETURN_TYPE] boolean   [VARIABLES] boolean  itsIgnoreDynamicScope  itsNeedsActivation  String  functionName  name  long  serialVersionUID  int  FUNCTION_EXPRESSION  FUNCTION_EXPRESSION_STATEMENT  FUNCTION_STATEMENT  charno  itsFunctionType  lineno  
[P5_Replace_Variable]^return itsNeedsActivation;^65^^^^^64^66^return itsIgnoreDynamicScope;^[CLASS] FunctionNode  [METHOD] getIgnoreDynamicScope [RETURN_TYPE] boolean   [VARIABLES] boolean  itsIgnoreDynamicScope  itsNeedsActivation  String  functionName  name  long  serialVersionUID  int  FUNCTION_EXPRESSION  FUNCTION_EXPRESSION_STATEMENT  FUNCTION_STATEMENT  charno  itsFunctionType  lineno  
[P8_Replace_Mix]^return lineno;^88^^^^^87^89^return itsFunctionType;^[CLASS] FunctionNode  [METHOD] getFunctionType [RETURN_TYPE] int   [VARIABLES] boolean  itsIgnoreDynamicScope  itsNeedsActivation  String  functionName  name  long  serialVersionUID  int  FUNCTION_EXPRESSION  FUNCTION_EXPRESSION_STATEMENT  FUNCTION_STATEMENT  charno  itsFunctionType  lineno  
