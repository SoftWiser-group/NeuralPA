[BugLab_Variable_Misuse]^_field = _field;^42^^^^^39^43^_field = field;^[CLASS] AnnotatedField Serialization  [METHOD] <init> [RETURN_TYPE] AnnotationMap)   Field field AnnotationMap annMap [VARIABLES] Serialization  _serialization  Field  _field  field  boolean  AnnotationMap  annMap  Class  clazz  String  name  long  serialVersionUID  
[BugLab_Variable_Misuse]^_serialization = _serialization;^57^^^^^53^58^_serialization = ser;^[CLASS] AnnotatedField Serialization  [METHOD] <init> [RETURN_TYPE] AnnotatedField$Serialization)   Serialization ser [VARIABLES] Serialization  _serialization  ser  Field  _field  field  Class  clazz  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^clazz = field.getDeclaringClass (  ) ;^181^^^^^180^184^clazz = f.getDeclaringClass (  ) ;^[CLASS] AnnotatedField Serialization  [METHOD] <init> [RETURN_TYPE] Field)   Field f [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  Class  clazz  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^name = field.getName (  ) ;^182^^^^^180^184^name = f.getName (  ) ;^[CLASS] AnnotatedField Serialization  [METHOD] <init> [RETURN_TYPE] Field)   Field f [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  Class  clazz  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return new AnnotatedField ( field, ann ) ;^47^^^^^46^48^return new AnnotatedField ( _field, ann ) ;^[CLASS] AnnotatedField Serialization  [METHOD] withAnnotations [RETURN_TYPE] AnnotatedField   AnnotationMap ann [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  AnnotationMap  ann  Class  clazz  String  name  long  serialVersionUID  
[BugLab_Argument_Swapping]^return new AnnotatedField ( ann, _field ) ;^47^^^^^46^48^return new AnnotatedField ( _field, ann ) ;^[CLASS] AnnotatedField Serialization  [METHOD] withAnnotations [RETURN_TYPE] AnnotatedField   AnnotationMap ann [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  AnnotationMap  ann  Class  clazz  String  name  long  serialVersionUID  
[BugLab_Variable_Misuse]^public Field getAnnotated (  )  { return field; }^67^^^^^62^72^public Field getAnnotated (  )  { return _field; }^[CLASS] AnnotatedField Serialization  [METHOD] getAnnotated [RETURN_TYPE] Field   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  Class  clazz  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^public int getModifiers (  )  { return field.getModifiers (  ) ; }^70^^^^^65^75^public int getModifiers (  )  { return _field.getModifiers (  ) ; }^[CLASS] AnnotatedField Serialization  [METHOD] getModifiers [RETURN_TYPE] int   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  Class  clazz  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^public String getName (  )  { return field.getName (  ) ; }^73^^^^^68^78^public String getName (  )  { return _field.getName (  ) ; }^[CLASS] AnnotatedField Serialization  [METHOD] getName [RETURN_TYPE] String   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  Class  clazz  String  name  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^return  ( acls == null )  ? null : _annotations.get ( _annotations ) ;^78^^^^^76^79^return  ( _annotations == null )  ? null : _annotations.get ( acls ) ;^[CLASS] AnnotatedField Serialization  [METHOD] getAnnotation [RETURN_TYPE] <A   Class<A> acls [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  Class  acls  clazz  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^return  ( _annotations != null )  ? null : _annotations.get ( acls ) ;^78^^^^^76^79^return  ( _annotations == null )  ? null : _annotations.get ( acls ) ;^[CLASS] AnnotatedField Serialization  [METHOD] getAnnotation [RETURN_TYPE] <A   Class<A> acls [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  Class  acls  clazz  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return field.getGenericType (  ) ;^83^^^^^82^84^return _field.getGenericType (  ) ;^[CLASS] AnnotatedField Serialization  [METHOD] getGenericType [RETURN_TYPE] Type   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  Class  acls  clazz  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return field.getType (  ) ;^88^^^^^87^89^return _field.getType (  ) ;^[CLASS] AnnotatedField Serialization  [METHOD] getRawType [RETURN_TYPE] Class   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  Class  acls  clazz  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^public Class<?> getDeclaringClass (  )  { return field.getDeclaringClass (  ) ; }^98^^^^^93^103^public Class<?> getDeclaringClass (  )  { return _field.getDeclaringClass (  ) ; }^[CLASS] AnnotatedField Serialization  [METHOD] getDeclaringClass [RETURN_TYPE] Class   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  Class  acls  clazz  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^public Member getMember (  )  { return field; }^101^^^^^96^106^public Member getMember (  )  { return _field; }^[CLASS] AnnotatedField Serialization  [METHOD] getMember [RETURN_TYPE] Member   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  Class  acls  clazz  String  name  boolean  long  serialVersionUID  
[BugLab_Argument_Swapping]^_field.set ( value, pojo ) ;^107^^^^^104^112^_field.set ( pojo, value ) ;^[CLASS] AnnotatedField Serialization  [METHOD] setValue [RETURN_TYPE] void   Object pojo Object value [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  Object  pojo  value  String  name  long  serialVersionUID  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException   instanceof   (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^109^110^^^^104^112^throw new IllegalArgumentException  (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^[CLASS] AnnotatedField Serialization  [METHOD] setValue [RETURN_TYPE] void   Object pojo Object value [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  Object  pojo  value  String  name  long  serialVersionUID  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException  !=  (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^109^110^^^^104^112^throw new IllegalArgumentException  (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^[CLASS] AnnotatedField Serialization  [METHOD] setValue [RETURN_TYPE] void   Object pojo Object value [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  Object  pojo  value  String  name  long  serialVersionUID  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException  |  (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^109^110^^^^104^112^throw new IllegalArgumentException  (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^[CLASS] AnnotatedField Serialization  [METHOD] setValue [RETURN_TYPE] void   Object pojo Object value [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  Object  pojo  value  String  name  long  serialVersionUID  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException  >  (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^109^110^^^^104^112^throw new IllegalArgumentException  (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^[CLASS] AnnotatedField Serialization  [METHOD] setValue [RETURN_TYPE] void   Object pojo Object value [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  Object  pojo  value  String  name  long  serialVersionUID  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException  ||  (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^109^110^^^^104^112^throw new IllegalArgumentException  (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^[CLASS] AnnotatedField Serialization  [METHOD] setValue [RETURN_TYPE] void   Object pojo Object value [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  Object  pojo  value  String  name  long  serialVersionUID  IllegalAccessException  e  
[BugLab_Variable_Misuse]^return field.get ( pojo ) ;^118^^^^^115^123^return _field.get ( pojo ) ;^[CLASS] AnnotatedField Serialization  [METHOD] getValue [RETURN_TYPE] Object   Object pojo [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  Object  pojo  String  name  long  serialVersionUID  IllegalAccessException  e  
[BugLab_Argument_Swapping]^return pojo.get ( _field ) ;^118^^^^^115^123^return _field.get ( pojo ) ;^[CLASS] AnnotatedField Serialization  [METHOD] getValue [RETURN_TYPE] Object   Object pojo [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  Object  pojo  String  name  long  serialVersionUID  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException  ||  (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^120^121^^^^115^123^throw new IllegalArgumentException  (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^[CLASS] AnnotatedField Serialization  [METHOD] getValue [RETURN_TYPE] Object   Object pojo [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  Object  pojo  String  name  long  serialVersionUID  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException  |  (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^120^121^^^^115^123^throw new IllegalArgumentException  (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^[CLASS] AnnotatedField Serialization  [METHOD] getValue [RETURN_TYPE] Object   Object pojo [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  Object  pojo  String  name  long  serialVersionUID  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException  <  (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^120^121^^^^115^123^throw new IllegalArgumentException  (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^[CLASS] AnnotatedField Serialization  [METHOD] getValue [RETURN_TYPE] Object   Object pojo [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  Object  pojo  String  name  long  serialVersionUID  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException  ^  (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^120^121^^^^115^123^throw new IllegalArgumentException  (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^[CLASS] AnnotatedField Serialization  [METHOD] getValue [RETURN_TYPE] Object   Object pojo [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  Object  pojo  String  name  long  serialVersionUID  IllegalAccessException  e  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException  &  (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^120^121^^^^115^123^throw new IllegalArgumentException  (" ")   for field " +getFullName (  ) +": "+e.getMessage (  ) , e ) ;^[CLASS] AnnotatedField Serialization  [METHOD] getValue [RETURN_TYPE] Object   Object pojo [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  Object  pojo  String  name  long  serialVersionUID  IllegalAccessException  e  
[BugLab_Wrong_Operator]^return getDeclaringClass (  &  ) .getName (  )  + "#" + getName (  ) ;^132^^^^^131^133^return getDeclaringClass (  ) .getName (  )  + "#" + getName (  ) ;^[CLASS] AnnotatedField Serialization  [METHOD] getFullName [RETURN_TYPE] String   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  Class  acls  clazz  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^return getDeclaringClass (  ) .getName (  )   &  "#" + getName (  ) ;^132^^^^^131^133^return getDeclaringClass (  ) .getName (  )  + "#" + getName (  ) ;^[CLASS] AnnotatedField Serialization  [METHOD] getFullName [RETURN_TYPE] String   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  Class  acls  clazz  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^return "[field "+getFullName (  ||  ) +"]";^140^^^^^138^141^return "[field "+getFullName (  ) +"]";^[CLASS] AnnotatedField Serialization  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  Class  acls  clazz  String  name  boolean  long  serialVersionUID  
[BugLab_Wrong_Operator]^return "[field "+getFullName (  <<  ) +"]";^140^^^^^138^141^return "[field "+getFullName (  ) +"]";^[CLASS] AnnotatedField Serialization  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  Class  acls  clazz  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^return new AnnotatedField ( new Serialization ( field )  ) ;^150^^^^^149^151^return new AnnotatedField ( new Serialization ( _field )  ) ;^[CLASS] AnnotatedField Serialization  [METHOD] writeReplace [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  Class  acls  clazz  String  name  boolean  long  serialVersionUID  
[BugLab_Variable_Misuse]^Class<?> clazz = ser.clazz;^154^^^^^153^166^Class<?> clazz = _serialization.clazz;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[BugLab_Argument_Swapping]^Class<?> clazz = _serialization;^154^^^^^153^166^Class<?> clazz = _serialization.clazz;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[BugLab_Argument_Swapping]^Class<?> clazz = _serialization.clazz.clazz;^154^^^^^153^166^Class<?> clazz = _serialization.clazz;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[BugLab_Variable_Misuse]^ifield  ( !f.isAccessible (  )  )  {^158^^^^^153^166^if  ( !f.isAccessible (  )  )  {^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[BugLab_Variable_Misuse]^ClassUtil.checkAndFixAccess ( field ) ;^159^^^^^153^166^ClassUtil.checkAndFixAccess ( f ) ;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[BugLab_Variable_Misuse]^return new AnnotatedField ( field, null ) ;^161^^^^^153^166^return new AnnotatedField ( f, null ) ;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[BugLab_Variable_Misuse]^Field f = clazz.getDeclaredField ( ser.name ) ;^156^^^^^153^166^Field f = clazz.getDeclaredField ( _serialization.name ) ;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[BugLab_Argument_Swapping]^Field f = _serialization.getDeclaredField ( clazz.name ) ;^156^^^^^153^166^Field f = clazz.getDeclaredField ( _serialization.name ) ;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[BugLab_Argument_Swapping]^Field f = clazz.getDeclaredField ( _serialization ) ;^156^^^^^153^166^Field f = clazz.getDeclaredField ( _serialization.name ) ;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[BugLab_Variable_Misuse]^Field f = clazz.getDeclaredField ( name ) ;^156^^^^^153^166^Field f = clazz.getDeclaredField ( _serialization.name ) ;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[BugLab_Argument_Swapping]^Field f = _serialization.name.getDeclaredField ( clazz ) ;^156^^^^^153^166^Field f = clazz.getDeclaredField ( _serialization.name ) ;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[BugLab_Variable_Misuse]^throw new IllegalArgumentException ( "Could not find method '"+name +"' from Class '"+clazz.getName (  )  ) ;^163^164^^^^153^166^throw new IllegalArgumentException ( "Could not find method '"+_serialization.name +"' from Class '"+clazz.getName (  )  ) ;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Could not find method '"+_serialization.name +"' from Class '"+clazz.getName (   instanceof   )  ) ;^163^164^^^^153^166^throw new IllegalArgumentException ( "Could not find method '"+_serialization.name +"' from Class '"+clazz.getName (  )  ) ;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Could not find method '"+_serialization.name +"' from Class '"+clazz.getName (  <  )  ) ;^163^164^^^^153^166^throw new IllegalArgumentException ( "Could not find method '"+_serialization.name +"' from Class '"+clazz.getName (  )  ) ;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[BugLab_Argument_Swapping]^Field f = clazz.getDeclaredField ( _serialization.name.name ) ;^156^^^^^153^166^Field f = clazz.getDeclaredField ( _serialization.name ) ;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Could not find method '"+_serialization.name +"' from Class '"+clazz.getName (  !=  )  ) ;^163^164^^^^153^166^throw new IllegalArgumentException ( "Could not find method '"+_serialization.name +"' from Class '"+clazz.getName (  )  ) ;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
[BugLab_Wrong_Operator]^throw new IllegalArgumentException ( "Could not find method '"+_serialization.name +"' from Class '"+clazz.getName (  >=  )  ) ;^163^164^^^^153^166^throw new IllegalArgumentException ( "Could not find method '"+_serialization.name +"' from Class '"+clazz.getName (  )  ) ;^[CLASS] AnnotatedField Serialization  [METHOD] readResolve [RETURN_TYPE] Object   [VARIABLES] Serialization  _serialization  ser  Field  _field  f  field  boolean  Class  acls  clazz  String  name  long  serialVersionUID  Exception  e  
