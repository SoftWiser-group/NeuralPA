[BugLab_Argument_Swapping]^String str = okNameForIsGetter ( name, am ) ;^20^^^^^17^25^String str = okNameForIsGetter ( am, name ) ;^[CLASS] BeanUtil  [METHOD] okNameForGetter [RETURN_TYPE] String   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  String  name  str  
[BugLab_Variable_Misuse]^String str = okNameForIsGetter ( am, str ) ;^20^^^^^17^25^String str = okNameForIsGetter ( am, name ) ;^[CLASS] BeanUtil  [METHOD] okNameForGetter [RETURN_TYPE] String   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  String  name  str  
[BugLab_Variable_Misuse]^if  ( name == null )  {^21^^^^^17^25^if  ( str == null )  {^[CLASS] BeanUtil  [METHOD] okNameForGetter [RETURN_TYPE] String   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  String  name  str  
[BugLab_Wrong_Operator]^if  ( str != null )  {^21^^^^^17^25^if  ( str == null )  {^[CLASS] BeanUtil  [METHOD] okNameForGetter [RETURN_TYPE] String   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  String  name  str  
[BugLab_Variable_Misuse]^str = okNameForRegularGetter ( am, str ) ;^22^^^^^17^25^str = okNameForRegularGetter ( am, name ) ;^[CLASS] BeanUtil  [METHOD] okNameForGetter [RETURN_TYPE] String   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  String  name  str  
[BugLab_Argument_Swapping]^str = okNameForRegularGetter ( name, am ) ;^22^^^^^17^25^str = okNameForRegularGetter ( am, name ) ;^[CLASS] BeanUtil  [METHOD] okNameForGetter [RETURN_TYPE] String   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  String  name  str  
[BugLab_Variable_Misuse]^return name;^24^^^^^17^25^return str;^[CLASS] BeanUtil  [METHOD] okNameForGetter [RETURN_TYPE] String   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  String  name  str  
[BugLab_Wrong_Literal]^return manglePropertyName ( name.substring ( 4 )  ) ;^49^^^^^27^52^return manglePropertyName ( name.substring ( 3 )  ) ;^[CLASS] BeanUtil  [METHOD] okNameForRegularGetter [RETURN_TYPE] String   AnnotatedMethod am String name [VARIABLES] boolean  AnnotatedMethod  am  String  name  
[BugLab_Wrong_Operator]^if  ( rt != Boolean.class || rt != Boolean.TYPE )  {^59^^^^^54^66^if  ( rt != Boolean.class && rt != Boolean.TYPE )  {^[CLASS] BeanUtil  [METHOD] okNameForIsGetter [RETURN_TYPE] String   AnnotatedMethod am String name [VARIABLES] boolean  AnnotatedMethod  am  Class  rt  String  name  
[BugLab_Wrong_Operator]^if  ( rt <= Boolean.class && rt != Boolean.TYPE )  {^59^^^^^54^66^if  ( rt != Boolean.class && rt != Boolean.TYPE )  {^[CLASS] BeanUtil  [METHOD] okNameForIsGetter [RETURN_TYPE] String   AnnotatedMethod am String name [VARIABLES] boolean  AnnotatedMethod  am  Class  rt  String  name  
[BugLab_Wrong_Operator]^if  ( rt == Boolean.class && rt != Boolean.TYPE )  {^59^^^^^54^66^if  ( rt != Boolean.class && rt != Boolean.TYPE )  {^[CLASS] BeanUtil  [METHOD] okNameForIsGetter [RETURN_TYPE] String   AnnotatedMethod am String name [VARIABLES] boolean  AnnotatedMethod  am  Class  rt  String  name  
[BugLab_Wrong_Literal]^return manglePropertyName ( name.substring (  )  ) ;^62^^^^^54^66^return manglePropertyName ( name.substring ( 2 )  ) ;^[CLASS] BeanUtil  [METHOD] okNameForIsGetter [RETURN_TYPE] String   AnnotatedMethod am String name [VARIABLES] boolean  AnnotatedMethod  am  Class  rt  String  name  
[BugLab_Wrong_Literal]^return manglePropertyName ( name.substring ( 1 )  ) ;^62^^^^^54^66^return manglePropertyName ( name.substring ( 2 )  ) ;^[CLASS] BeanUtil  [METHOD] okNameForIsGetter [RETURN_TYPE] String   AnnotatedMethod am String name [VARIABLES] boolean  AnnotatedMethod  am  Class  rt  String  name  
[BugLab_Wrong_Operator]^if  ( rt > Boolean.class && rt != Boolean.TYPE )  {^59^^^^^54^66^if  ( rt != Boolean.class && rt != Boolean.TYPE )  {^[CLASS] BeanUtil  [METHOD] okNameForIsGetter [RETURN_TYPE] String   AnnotatedMethod am String name [VARIABLES] boolean  AnnotatedMethod  am  Class  rt  String  name  
[BugLab_Wrong_Literal]^return manglePropertyName ( name.substring ( 3 )  ) ;^62^^^^^54^66^return manglePropertyName ( name.substring ( 2 )  ) ;^[CLASS] BeanUtil  [METHOD] okNameForIsGetter [RETURN_TYPE] String   AnnotatedMethod am String name [VARIABLES] boolean  AnnotatedMethod  am  Class  rt  String  name  
[BugLab_Wrong_Operator]^if  ( name == null )  {^71^^^^^68^81^if  ( name != null )  {^[CLASS] BeanUtil  [METHOD] okNameForSetter [RETURN_TYPE] String   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  String  name  
[BugLab_Argument_Swapping]^if  ( prefix.startsWith ( name )  )  {^86^^^^^83^90^if  ( name.startsWith ( prefix )  )  {^[CLASS] BeanUtil  [METHOD] okNameForMutator [RETURN_TYPE] String   AnnotatedMethod am String prefix [VARIABLES] boolean  AnnotatedMethod  am  String  name  prefix  
[BugLab_Argument_Swapping]^return manglePropertyName ( prefix.substring ( name.length (  )  )  ) ;^87^^^^^83^90^return manglePropertyName ( name.substring ( prefix.length (  )  )  ) ;^[CLASS] BeanUtil  [METHOD] okNameForMutator [RETURN_TYPE] String   AnnotatedMethod am String prefix [VARIABLES] boolean  AnnotatedMethod  am  String  name  prefix  
[BugLab_Variable_Misuse]^return manglePropertyName ( name.substring ( name.length (  )  )  ) ;^87^^^^^83^90^return manglePropertyName ( name.substring ( prefix.length (  )  )  ) ;^[CLASS] BeanUtil  [METHOD] okNameForMutator [RETURN_TYPE] String   AnnotatedMethod am String prefix [VARIABLES] boolean  AnnotatedMethod  am  String  name  prefix  
[BugLab_Wrong_Operator]^if  ( rt == null && !rt.isArray (  )  )  {^113^^^^^109^132^if  ( rt == null || !rt.isArray (  )  )  {^[CLASS] BeanUtil  [METHOD] isCglibGetCallbacks [RETURN_TYPE] boolean   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  Class  compType  rt  Package  pkg  String  pname  
[BugLab_Wrong_Operator]^if  ( rt != null || !rt.isArray (  )  )  {^113^^^^^109^132^if  ( rt == null || !rt.isArray (  )  )  {^[CLASS] BeanUtil  [METHOD] isCglibGetCallbacks [RETURN_TYPE] boolean   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  Class  compType  rt  Package  pkg  String  pname  
[BugLab_Wrong_Literal]^return true;^114^^^^^109^132^return false;^[CLASS] BeanUtil  [METHOD] isCglibGetCallbacks [RETURN_TYPE] boolean   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  Class  compType  rt  Package  pkg  String  pname  
[BugLab_Wrong_Operator]^if  ( pkg == null )  {^123^^^^^109^132^if  ( pkg != null )  {^[CLASS] BeanUtil  [METHOD] isCglibGetCallbacks [RETURN_TYPE] boolean   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  Class  compType  rt  Package  pkg  String  pname  
[BugLab_Wrong_Operator]^if  ( pname.startsWith ( "net.sf.cglib" )  && pname.startsWith ( "org.hibernate.repackage.cglib" )  )  {^125^126^127^^^109^132^if  ( pname.startsWith ( "net.sf.cglib" )  || pname.startsWith ( "org.hibernate.repackage.cglib" )  )  {^[CLASS] BeanUtil  [METHOD] isCglibGetCallbacks [RETURN_TYPE] boolean   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  Class  compType  rt  Package  pkg  String  pname  
[BugLab_Wrong_Literal]^return false;^128^^^^^109^132^return true;^[CLASS] BeanUtil  [METHOD] isCglibGetCallbacks [RETURN_TYPE] boolean   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  Class  compType  rt  Package  pkg  String  pname  
[BugLab_Wrong_Literal]^return true;^131^^^^^109^132^return false;^[CLASS] BeanUtil  [METHOD] isCglibGetCallbacks [RETURN_TYPE] boolean   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  Class  compType  rt  Package  pkg  String  pname  
[BugLab_Wrong_Literal]^Class<?> argType = am.getRawParameterType (  ) ;^140^^^^^138^146^Class<?> argType = am.getRawParameterType ( 0 ) ;^[CLASS] BeanUtil  [METHOD] isGroovyMetaClassSetter [RETURN_TYPE] boolean   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  Class  argType  Package  pkg  
[BugLab_Wrong_Operator]^if  ( pkg != null || pkg.getName (  ) .startsWith ( "groovy.lang" )  )  {^142^^^^^138^146^if  ( pkg != null && pkg.getName (  ) .startsWith ( "groovy.lang" )  )  {^[CLASS] BeanUtil  [METHOD] isGroovyMetaClassSetter [RETURN_TYPE] boolean   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  Class  argType  Package  pkg  
[BugLab_Wrong_Operator]^if  ( pkg == null && pkg.getName (  ) .startsWith ( "groovy.lang" )  )  {^142^^^^^138^146^if  ( pkg != null && pkg.getName (  ) .startsWith ( "groovy.lang" )  )  {^[CLASS] BeanUtil  [METHOD] isGroovyMetaClassSetter [RETURN_TYPE] boolean   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  Class  argType  Package  pkg  
[BugLab_Wrong_Literal]^return false;^143^^^^^138^146^return true;^[CLASS] BeanUtil  [METHOD] isGroovyMetaClassSetter [RETURN_TYPE] boolean   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  Class  argType  Package  pkg  
[BugLab_Wrong_Literal]^return true;^145^^^^^138^146^return false;^[CLASS] BeanUtil  [METHOD] isGroovyMetaClassSetter [RETURN_TYPE] boolean   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  Class  argType  Package  pkg  
[BugLab_Wrong_Operator]^if  ( rt == null && rt.isArray (  )  )  {^154^^^^^151^162^if  ( rt == null || rt.isArray (  )  )  {^[CLASS] BeanUtil  [METHOD] isGroovyMetaClassGetter [RETURN_TYPE] boolean   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  Class  rt  Package  pkg  
[BugLab_Wrong_Operator]^if  ( rt != null || rt.isArray (  )  )  {^154^^^^^151^162^if  ( rt == null || rt.isArray (  )  )  {^[CLASS] BeanUtil  [METHOD] isGroovyMetaClassGetter [RETURN_TYPE] boolean   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  Class  rt  Package  pkg  
[BugLab_Wrong_Literal]^return true;^155^^^^^151^162^return false;^[CLASS] BeanUtil  [METHOD] isGroovyMetaClassGetter [RETURN_TYPE] boolean   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  Class  rt  Package  pkg  
[BugLab_Wrong_Operator]^if  ( pkg != null || pkg.getName (  ) .startsWith ( "groovy.lang" )  )  {^158^^^^^151^162^if  ( pkg != null && pkg.getName (  ) .startsWith ( "groovy.lang" )  )  {^[CLASS] BeanUtil  [METHOD] isGroovyMetaClassGetter [RETURN_TYPE] boolean   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  Class  rt  Package  pkg  
[BugLab_Wrong_Operator]^if  ( pkg == null && pkg.getName (  ) .startsWith ( "groovy.lang" )  )  {^158^^^^^151^162^if  ( pkg != null && pkg.getName (  ) .startsWith ( "groovy.lang" )  )  {^[CLASS] BeanUtil  [METHOD] isGroovyMetaClassGetter [RETURN_TYPE] boolean   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  Class  rt  Package  pkg  
[BugLab_Wrong_Literal]^return false;^159^^^^^151^162^return true;^[CLASS] BeanUtil  [METHOD] isGroovyMetaClassGetter [RETURN_TYPE] boolean   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  Class  rt  Package  pkg  
[BugLab_Wrong_Literal]^return true;^161^^^^^151^162^return false;^[CLASS] BeanUtil  [METHOD] isGroovyMetaClassGetter [RETURN_TYPE] boolean   AnnotatedMethod am [VARIABLES] boolean  AnnotatedMethod  am  Class  rt  Package  pkg  
[BugLab_Variable_Misuse]^if  ( i == 0 )  {^176^^^^^171^193^if  ( len == 0 )  {^[CLASS] BeanUtil  [METHOD] manglePropertyName [RETURN_TYPE] String   String basename [VARIABLES] boolean  StringBuilder  sb  char  lower  upper  String  basename  int  i  len  
[BugLab_Wrong_Operator]^if  ( len != 0 )  {^176^^^^^171^193^if  ( len == 0 )  {^[CLASS] BeanUtil  [METHOD] manglePropertyName [RETURN_TYPE] String   String basename [VARIABLES] boolean  StringBuilder  sb  char  lower  upper  String  basename  int  i  len  
[BugLab_Wrong_Literal]^if  ( len == len )  {^176^^^^^171^193^if  ( len == 0 )  {^[CLASS] BeanUtil  [METHOD] manglePropertyName [RETURN_TYPE] String   String basename [VARIABLES] boolean  StringBuilder  sb  char  lower  upper  String  basename  int  i  len  
[BugLab_Argument_Swapping]^if  ( lower == upper )  {^184^^^^^171^193^if  ( upper == lower )  {^[CLASS] BeanUtil  [METHOD] manglePropertyName [RETURN_TYPE] String   String basename [VARIABLES] boolean  StringBuilder  sb  char  lower  upper  String  basename  int  i  len  
[BugLab_Wrong_Operator]^if  ( upper > lower )  {^184^^^^^171^193^if  ( upper == lower )  {^[CLASS] BeanUtil  [METHOD] manglePropertyName [RETURN_TYPE] String   String basename [VARIABLES] boolean  StringBuilder  sb  char  lower  upper  String  basename  int  i  len  
[BugLab_Wrong_Operator]^if  ( sb != null )  {^187^^^^^171^193^if  ( sb == null )  {^[CLASS] BeanUtil  [METHOD] manglePropertyName [RETURN_TYPE] String   String basename [VARIABLES] boolean  StringBuilder  sb  char  lower  upper  String  basename  int  i  len  
[BugLab_Variable_Misuse]^for  ( lennt i = 0; i < len; ++i )  {^181^^^^^171^193^for  ( int i = 0; i < len; ++i )  {^[CLASS] BeanUtil  [METHOD] manglePropertyName [RETURN_TYPE] String   String basename [VARIABLES] boolean  StringBuilder  sb  char  lower  upper  String  basename  int  i  len  
[BugLab_Argument_Swapping]^for  ( lennt i = 0; i < i; ++i )  {^181^^^^^171^193^for  ( int i = 0; i < len; ++i )  {^[CLASS] BeanUtil  [METHOD] manglePropertyName [RETURN_TYPE] String   String basename [VARIABLES] boolean  StringBuilder  sb  char  lower  upper  String  basename  int  i  len  
[BugLab_Wrong_Operator]^for  ( int i = 0; i == len; ++i )  {^181^^^^^171^193^for  ( int i = 0; i < len; ++i )  {^[CLASS] BeanUtil  [METHOD] manglePropertyName [RETURN_TYPE] String   String basename [VARIABLES] boolean  StringBuilder  sb  char  lower  upper  String  basename  int  i  len  
[BugLab_Wrong_Literal]^for  ( int i = len; i < len; ++i )  {^181^^^^^171^193^for  ( int i = 0; i < len; ++i )  {^[CLASS] BeanUtil  [METHOD] manglePropertyName [RETURN_TYPE] String   String basename [VARIABLES] boolean  StringBuilder  sb  char  lower  upper  String  basename  int  i  len  
[BugLab_Wrong_Operator]^if  ( upper <= lower )  {^184^^^^^171^193^if  ( upper == lower )  {^[CLASS] BeanUtil  [METHOD] manglePropertyName [RETURN_TYPE] String   String basename [VARIABLES] boolean  StringBuilder  sb  char  lower  upper  String  basename  int  i  len  
[BugLab_Variable_Misuse]^char upper = basename.charAt ( len ) ;^182^^^^^171^193^char upper = basename.charAt ( i ) ;^[CLASS] BeanUtil  [METHOD] manglePropertyName [RETURN_TYPE] String   String basename [VARIABLES] boolean  StringBuilder  sb  char  lower  upper  String  basename  int  i  len  
[BugLab_Argument_Swapping]^char upper = i.charAt ( basename ) ;^182^^^^^171^193^char upper = basename.charAt ( i ) ;^[CLASS] BeanUtil  [METHOD] manglePropertyName [RETURN_TYPE] String   String basename [VARIABLES] boolean  StringBuilder  sb  char  lower  upper  String  basename  int  i  len  
[BugLab_Variable_Misuse]^char lower = Character.toLowerCase ( lower ) ;^183^^^^^171^193^char lower = Character.toLowerCase ( upper ) ;^[CLASS] BeanUtil  [METHOD] manglePropertyName [RETURN_TYPE] String   String basename [VARIABLES] boolean  StringBuilder  sb  char  lower  upper  String  basename  int  i  len  
[BugLab_Variable_Misuse]^sb.setCharAt ( len, lower ) ;^190^^^^^171^193^sb.setCharAt ( i, lower ) ;^[CLASS] BeanUtil  [METHOD] manglePropertyName [RETURN_TYPE] String   String basename [VARIABLES] boolean  StringBuilder  sb  char  lower  upper  String  basename  int  i  len  
[BugLab_Variable_Misuse]^sb.setCharAt ( i, upper ) ;^190^^^^^171^193^sb.setCharAt ( i, lower ) ;^[CLASS] BeanUtil  [METHOD] manglePropertyName [RETURN_TYPE] String   String basename [VARIABLES] boolean  StringBuilder  sb  char  lower  upper  String  basename  int  i  len  
[BugLab_Argument_Swapping]^sb.setCharAt ( lower, i ) ;^190^^^^^171^193^sb.setCharAt ( i, lower ) ;^[CLASS] BeanUtil  [METHOD] manglePropertyName [RETURN_TYPE] String   String basename [VARIABLES] boolean  StringBuilder  sb  char  lower  upper  String  basename  int  i  len  
[BugLab_Argument_Swapping]^return  ( basename == null )  ? sb : sb.toString (  ) ;^192^^^^^171^193^return  ( sb == null )  ? basename : sb.toString (  ) ;^[CLASS] BeanUtil  [METHOD] manglePropertyName [RETURN_TYPE] String   String basename [VARIABLES] boolean  StringBuilder  sb  char  lower  upper  String  basename  int  i  len  
[BugLab_Wrong_Operator]^return  ( sb != null )  ? basename : sb.toString (  ) ;^192^^^^^171^193^return  ( sb == null )  ? basename : sb.toString (  ) ;^[CLASS] BeanUtil  [METHOD] manglePropertyName [RETURN_TYPE] String   String basename [VARIABLES] boolean  StringBuilder  sb  char  lower  upper  String  basename  int  i  len  
