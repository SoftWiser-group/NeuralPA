[BugLab_Variable_Misuse]^if  ( _serialized == null )  {^129^^^^^127^143^if  ( token == null )  {^[CLASS] JsonToken  [METHOD] <init> [RETURN_TYPE] String)   String token [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  int  i  len  char[]  _serializedChars  
[BugLab_Wrong_Operator]^if  ( token != null )  {^129^^^^^127^143^if  ( token == null )  {^[CLASS] JsonToken  [METHOD] <init> [RETURN_TYPE] String)   String token [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  int  i  len  char[]  _serializedChars  
[BugLab_Variable_Misuse]^for  ( lennt i = 0; i < len; ++i )  {^139^^^^^127^143^for  ( int i = 0; i < len; ++i )  {^[CLASS] JsonToken  [METHOD] <init> [RETURN_TYPE] String)   String token [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  int  i  len  char[]  _serializedChars  
[BugLab_Argument_Swapping]^for  ( lennt i = 0; i < i; ++i )  {^139^^^^^127^143^for  ( int i = 0; i < len; ++i )  {^[CLASS] JsonToken  [METHOD] <init> [RETURN_TYPE] String)   String token [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  int  i  len  char[]  _serializedChars  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= len; ++i )  {^139^^^^^127^143^for  ( int i = 0; i < len; ++i )  {^[CLASS] JsonToken  [METHOD] <init> [RETURN_TYPE] String)   String token [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  int  i  len  char[]  _serializedChars  
[BugLab_Wrong_Literal]^for  ( int i = i; i < len; ++i )  {^139^^^^^127^143^for  ( int i = 0; i < len; ++i )  {^[CLASS] JsonToken  [METHOD] <init> [RETURN_TYPE] String)   String token [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  int  i  len  char[]  _serializedChars  
[BugLab_Variable_Misuse]^_serialized = _serialized;^134^^^^^127^143^_serialized = token;^[CLASS] JsonToken  [METHOD] <init> [RETURN_TYPE] String)   String token [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  int  i  len  char[]  _serializedChars  
[BugLab_Variable_Misuse]^_serializedChars = _serialized.toCharArray (  ) ;^135^^^^^127^143^_serializedChars = token.toCharArray (  ) ;^[CLASS] JsonToken  [METHOD] <init> [RETURN_TYPE] String)   String token [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  int  i  len  char[]  _serializedChars  
[BugLab_Variable_Misuse]^int len = i;^137^^^^^127^143^int len = _serializedChars.length;^[CLASS] JsonToken  [METHOD] <init> [RETURN_TYPE] String)   String token [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  int  i  len  char[]  _serializedChars  
[BugLab_Argument_Swapping]^int len = _serializedChars;^137^^^^^127^143^int len = _serializedChars.length;^[CLASS] JsonToken  [METHOD] <init> [RETURN_TYPE] String)   String token [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  int  i  len  char[]  _serializedChars  
[BugLab_Argument_Swapping]^int len = _serializedChars.length.length;^137^^^^^127^143^int len = _serializedChars.length;^[CLASS] JsonToken  [METHOD] <init> [RETURN_TYPE] String)   String token [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  int  i  len  char[]  _serializedChars  
[BugLab_Wrong_Literal]^for  ( int i = len; i < len; ++i )  {^139^^^^^127^143^for  ( int i = 0; i < len; ++i )  {^[CLASS] JsonToken  [METHOD] <init> [RETURN_TYPE] String)   String token [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  int  i  len  char[]  _serializedChars  
[BugLab_Wrong_Operator]^for  ( int i = 0; i > len; ++i )  {^139^^^^^127^143^for  ( int i = 0; i < len; ++i )  {^[CLASS] JsonToken  [METHOD] <init> [RETURN_TYPE] String)   String token [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  int  i  len  char[]  _serializedChars  
[BugLab_Variable_Misuse]^return  ( this == VALUE_FALSE )  ||  ( this == VALUE_NUMBER_FLOAT ) ;^150^^^^^149^151^return  ( this == VALUE_NUMBER_INT )  ||  ( this == VALUE_NUMBER_FLOAT ) ;^[CLASS] JsonToken  [METHOD] isNumeric [RETURN_TYPE] boolean   [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  char[]  _serializedChars  
[BugLab_Variable_Misuse]^return  ( this == VALUE_NUMBER_INT )  ||  ( this == VALUE_EMBEDDED_OBJECT ) ;^150^^^^^149^151^return  ( this == VALUE_NUMBER_INT )  ||  ( this == VALUE_NUMBER_FLOAT ) ;^[CLASS] JsonToken  [METHOD] isNumeric [RETURN_TYPE] boolean   [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  char[]  _serializedChars  
[BugLab_Argument_Swapping]^return  ( this == VALUE_NUMBER_FLOAT )  ||  ( this == VALUE_NUMBER_INT ) ;^150^^^^^149^151^return  ( this == VALUE_NUMBER_INT )  ||  ( this == VALUE_NUMBER_FLOAT ) ;^[CLASS] JsonToken  [METHOD] isNumeric [RETURN_TYPE] boolean   [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  char[]  _serializedChars  
[BugLab_Wrong_Operator]^return  ( this == VALUE_NUMBER_INT )  &&  ( this == VALUE_NUMBER_FLOAT ) ;^150^^^^^149^151^return  ( this == VALUE_NUMBER_INT )  ||  ( this == VALUE_NUMBER_FLOAT ) ;^[CLASS] JsonToken  [METHOD] isNumeric [RETURN_TYPE] boolean   [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  char[]  _serializedChars  
[BugLab_Wrong_Operator]^return  ( this >= VALUE_NUMBER_INT )  ||  ( this == VALUE_NUMBER_FLOAT ) ;^150^^^^^149^151^return  ( this == VALUE_NUMBER_INT )  ||  ( this == VALUE_NUMBER_FLOAT ) ;^[CLASS] JsonToken  [METHOD] isNumeric [RETURN_TYPE] boolean   [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  char[]  _serializedChars  
[BugLab_Wrong_Operator]^return  ( this == VALUE_NUMBER_INT )  ||  ( this != VALUE_NUMBER_FLOAT ) ;^150^^^^^149^151^return  ( this == VALUE_NUMBER_INT )  ||  ( this == VALUE_NUMBER_FLOAT ) ;^[CLASS] JsonToken  [METHOD] isNumeric [RETURN_TYPE] boolean   [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  char[]  _serializedChars  
[BugLab_Variable_Misuse]^return ordinal (  )  >= VALUE_TRUE.ordinal (  ) ;^160^^^^^158^161^return ordinal (  )  >= VALUE_EMBEDDED_OBJECT.ordinal (  ) ;^[CLASS] JsonToken  [METHOD] isScalarValue [RETURN_TYPE] boolean   [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  char[]  _serializedChars  
[BugLab_Wrong_Operator]^return ordinal (  )  > VALUE_EMBEDDED_OBJECT.ordinal (  ) ;^160^^^^^158^161^return ordinal (  )  >= VALUE_EMBEDDED_OBJECT.ordinal (  ) ;^[CLASS] JsonToken  [METHOD] isScalarValue [RETURN_TYPE] boolean   [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  char[]  _serializedChars  
[BugLab_Variable_Misuse]^return ordinal (  )  >= VALUE_NUMBER_FLOAT.ordinal (  ) ;^160^^^^^158^161^return ordinal (  )  >= VALUE_EMBEDDED_OBJECT.ordinal (  ) ;^[CLASS] JsonToken  [METHOD] isScalarValue [RETURN_TYPE] boolean   [VARIABLES] byte[]  _serializedBytes  JsonToken  END_ARRAY  END_OBJECT  FIELD_NAME  NOT_AVAILABLE  START_ARRAY  START_OBJECT  VALUE_EMBEDDED_OBJECT  VALUE_FALSE  VALUE_NULL  VALUE_NUMBER_FLOAT  VALUE_NUMBER_INT  VALUE_STRING  VALUE_TRUE  String  _serialized  token  boolean  char[]  _serializedChars  
