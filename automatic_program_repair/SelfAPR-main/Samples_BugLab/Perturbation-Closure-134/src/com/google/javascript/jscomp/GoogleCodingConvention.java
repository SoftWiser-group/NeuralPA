[BugLab_Variable_Misuse]^if  ( VAR_ARGS_NAME.length (  )  <= 1 )  {^61^^^^^60^81^if  ( name.length (  )  <= 1 )  {^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Operator]^if  ( name.length (  )  < 1 )  {^61^^^^^60^81^if  ( name.length (  )  <= 1 )  {^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Literal]^if  ( name.length (  )  <= pos )  {^61^^^^^60^81^if  ( name.length (  )  <= 1 )  {^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Literal]^return true;^62^^^^^60^81^return false;^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Variable_Misuse]^int pos = VAR_ARGS_NAME.lastIndexOf ( '$' ) ;^67^^^^^60^81^int pos = name.lastIndexOf ( '$' ) ;^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Operator]^if  ( pos > 0 )  {^68^^^^^60^81^if  ( pos >= 0 )  {^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Literal]^if  ( pos >=  )  {^68^^^^^60^81^if  ( pos >= 0 )  {^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Literal]^if  ( pos >= pos )  {^68^^^^^60^81^if  ( pos >= 0 )  {^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Variable_Misuse]^if  ( VAR_ARGS_NAME.length (  )  == 0 )  {^70^^^^^60^81^if  ( name.length (  )  == 0 )  {^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Operator]^if  ( name.length (  )  >= 0 )  {^70^^^^^60^81^if  ( name.length (  )  == 0 )  {^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Literal]^if  ( name.length (  )  == 1 )  {^70^^^^^60^81^if  ( name.length (  )  == 0 )  {^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Literal]^return true;^71^^^^^60^81^return false;^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Variable_Misuse]^name = VAR_ARGS_NAME.substring ( pos + 1 ) ;^69^^^^^60^81^name = name.substring ( pos + 1 ) ;^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Argument_Swapping]^name = pos.substring ( name + 1 ) ;^69^^^^^60^81^name = name.substring ( pos + 1 ) ;^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Operator]^name = name.substring ( pos  |  1 ) ;^69^^^^^60^81^name = name.substring ( pos + 1 ) ;^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Literal]^name = name.substring ( pos + pos ) ;^69^^^^^60^81^name = name.substring ( pos + 1 ) ;^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Literal]^name = name.substring ( pos  ) ;^69^^^^^60^81^name = name.substring ( pos + 1 ) ;^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Operator]^if  ( name.length (  )  != 0 )  {^70^^^^^60^81^if  ( name.length (  )  == 0 )  {^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Literal]^if  ( name.length (  )  == pos )  {^70^^^^^60^81^if  ( name.length (  )  == 0 )  {^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Operator]^name = name.substring ( pos  !=  1 ) ;^69^^^^^60^81^name = name.substring ( pos + 1 ) ;^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Operator]^name = name.substring ( pos  ||  1 ) ;^69^^^^^60^81^name = name.substring ( pos + 1 ) ;^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Literal]^name = name.substring ( pos +  ) ;^69^^^^^60^81^name = name.substring ( pos + 1 ) ;^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Variable_Misuse]^if  ( !Character.isUpperCase ( VAR_ARGS_NAME.charAt ( 0 )  )  )  {^75^^^^^60^81^if  ( !Character.isUpperCase ( name.charAt ( 0 )  )  )  {^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Literal]^return true;^76^^^^^60^81^return false;^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Literal]^if  ( !Character.isUpperCase ( name.charAt ( pos )  )  )  {^75^^^^^60^81^if  ( !Character.isUpperCase ( name.charAt ( 0 )  )  )  {^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Wrong_Literal]^if  ( !Character.isUpperCase ( name.charAt ( -1 )  )  )  {^75^^^^^60^81^if  ( !Character.isUpperCase ( name.charAt ( 0 )  )  )  {^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Variable_Misuse]^return VAR_ARGS_NAME.toUpperCase (  ) .equals ( name ) ;^80^^^^^60^81^return name.toUpperCase (  ) .equals ( name ) ;^[CLASS] GoogleCodingConvention  [METHOD] isConstant [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  name  boolean  int  pos  
[BugLab_Variable_Misuse]^return ENUM_KEY_PATTERN.matcher ( name ) .matches (  ) ;^98^^^^^97^99^return ENUM_KEY_PATTERN.matcher ( key ) .matches (  ) ;^[CLASS] GoogleCodingConvention  [METHOD] isValidEnumKey [RETURN_TYPE] boolean   String key [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  key  name  boolean  
[BugLab_Argument_Swapping]^return key.matcher ( ENUM_KEY_PATTERN ) .matches (  ) ;^98^^^^^97^99^return ENUM_KEY_PATTERN.matcher ( key ) .matches (  ) ;^[CLASS] GoogleCodingConvention  [METHOD] isValidEnumKey [RETURN_TYPE] boolean   String key [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  key  name  boolean  
[BugLab_Variable_Misuse]^return parameter.getString (  ) .startsWith ( name ) ;^109^^^^^108^110^return parameter.getString (  ) .startsWith ( OPTIONAL_ARG_PREFIX ) ;^[CLASS] GoogleCodingConvention  [METHOD] isOptionalParameter [RETURN_TYPE] boolean   Node parameter [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  key  name  boolean  Node  parameter  
[BugLab_Argument_Swapping]^return OPTIONAL_ARG_PREFIX.getString (  ) .startsWith ( parameter ) ;^109^^^^^108^110^return parameter.getString (  ) .startsWith ( OPTIONAL_ARG_PREFIX ) ;^[CLASS] GoogleCodingConvention  [METHOD] isOptionalParameter [RETURN_TYPE] boolean   Node parameter [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  key  name  boolean  Node  parameter  
[BugLab_Variable_Misuse]^return name.equals ( parameter.getString (  )  ) ;^114^^^^^113^115^return VAR_ARGS_NAME.equals ( parameter.getString (  )  ) ;^[CLASS] GoogleCodingConvention  [METHOD] isVarArgsParameter [RETURN_TYPE] boolean   Node parameter [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  key  name  boolean  Node  parameter  
[BugLab_Argument_Swapping]^return parameter.equals ( VAR_ARGS_NAME.getString (  )  ) ;^114^^^^^113^115^return VAR_ARGS_NAME.equals ( parameter.getString (  )  ) ;^[CLASS] GoogleCodingConvention  [METHOD] isVarArgsParameter [RETURN_TYPE] boolean   Node parameter [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  key  name  boolean  Node  parameter  
[BugLab_Variable_Misuse]^return !local && key.startsWith ( "_" ) ;^125^^^^^124^126^return !local && name.startsWith ( "_" ) ;^[CLASS] GoogleCodingConvention  [METHOD] isExported [RETURN_TYPE] boolean   String name boolean local [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  key  name  boolean  local  
[BugLab_Wrong_Operator]^return !local || name.startsWith ( "_" ) ;^125^^^^^124^126^return !local && name.startsWith ( "_" ) ;^[CLASS] GoogleCodingConvention  [METHOD] isExported [RETURN_TYPE] boolean   String name boolean local [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  key  name  boolean  local  
[BugLab_Variable_Misuse]^return key.endsWith ( "_" )  && !isExported ( name ) ;^136^^^^^135^137^return name.endsWith ( "_" )  && !isExported ( name ) ;^[CLASS] GoogleCodingConvention  [METHOD] isPrivate [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  key  name  boolean  
[BugLab_Wrong_Operator]^return name.endsWith ( "_" )  || !isExported ( name ) ;^136^^^^^135^137^return name.endsWith ( "_" )  && !isExported ( name ) ;^[CLASS] GoogleCodingConvention  [METHOD] isPrivate [RETURN_TYPE] boolean   String name [VARIABLES] Pattern  ENUM_KEY_PATTERN  String  OPTIONAL_ARG_PREFIX  VAR_ARGS_NAME  key  name  boolean  