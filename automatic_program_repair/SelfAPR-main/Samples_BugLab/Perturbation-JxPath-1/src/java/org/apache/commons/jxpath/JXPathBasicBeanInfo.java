[BugLab_Wrong_Literal]^private boolean atomic = true;^37^^^^^32^42^private boolean atomic = false;^[CLASS] JXPathBasicBeanInfo 1   [VARIABLES] 
[BugLab_Variable_Misuse]^this.clazz = dynamicPropertyHandlerClass;^44^^^^^43^45^this.clazz = clazz;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] <init> [RETURN_TYPE] Class)   Class clazz [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  propertyDescriptors  String[]  propertyNames  
[BugLab_Variable_Misuse]^this.clazz = dynamicPropertyHandlerClass;^48^^^^^47^50^this.clazz = clazz;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] <init> [RETURN_TYPE] Class,boolean)   Class clazz boolean atomic [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  propertyDescriptors  String[]  propertyNames  
[BugLab_Variable_Misuse]^this.clazz = dynamicPropertyHandlerClass;^53^^^^^52^56^this.clazz = clazz;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] <init> [RETURN_TYPE] Class)   Class clazz Class dynamicPropertyHandlerClass [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  propertyDescriptors  String[]  propertyNames  
[BugLab_Wrong_Literal]^this.atomic = true;^54^^^^^52^56^this.atomic = false;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] <init> [RETURN_TYPE] Class)   Class clazz Class dynamicPropertyHandlerClass [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  propertyDescriptors  String[]  propertyNames  
[BugLab_Variable_Misuse]^this.dynamicPropertyHandlerClass = clazz;^55^^^^^52^56^this.dynamicPropertyHandlerClass = dynamicPropertyHandlerClass;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] <init> [RETURN_TYPE] Class)   Class clazz Class dynamicPropertyHandlerClass [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  propertyDescriptors  String[]  propertyNames  
[BugLab_Variable_Misuse]^return clazz != null;^70^^^^^69^71^return dynamicPropertyHandlerClass != null;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] isDynamic [RETURN_TYPE] boolean   [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  propertyDescriptors  String[]  propertyNames  
[BugLab_Wrong_Operator]^return dynamicPropertyHandlerClass == null;^70^^^^^69^71^return dynamicPropertyHandlerClass != null;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] isDynamic [RETURN_TYPE] boolean   [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  propertyDescriptors  String[]  propertyNames  
[BugLab_Variable_Misuse]^if  ( pds == null )  {^74^^^^^73^99^if  ( propertyDescriptors == null )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptors [RETURN_TYPE] PropertyDescriptor[]   [VARIABLES] boolean  atomic  BeanInfo  bi  IntrospectionException  ex  Class  clazz  dynamicPropertyHandlerClass  Object  left  right  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  propertyNames  
[BugLab_Wrong_Operator]^if  ( propertyDescriptors != null )  {^74^^^^^73^99^if  ( propertyDescriptors == null )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptors [RETURN_TYPE] PropertyDescriptor[]   [VARIABLES] boolean  atomic  BeanInfo  bi  IntrospectionException  ex  Class  clazz  dynamicPropertyHandlerClass  Object  left  right  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  propertyNames  
[BugLab_Variable_Misuse]^if  ( dynamicPropertyHandlerClass.isInterface (  )  )  {^77^^^^^73^99^if  ( clazz.isInterface (  )  )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptors [RETURN_TYPE] PropertyDescriptor[]   [VARIABLES] boolean  atomic  BeanInfo  bi  IntrospectionException  ex  Class  clazz  dynamicPropertyHandlerClass  Object  left  right  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  propertyNames  
[BugLab_Variable_Misuse]^bi = Introspector.getBeanInfo ( dynamicPropertyHandlerClass, Object.class ) ;^81^^^^^77^82^bi = Introspector.getBeanInfo ( clazz, Object.class ) ;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptors [RETURN_TYPE] PropertyDescriptor[]   [VARIABLES] boolean  atomic  BeanInfo  bi  IntrospectionException  ex  Class  clazz  dynamicPropertyHandlerClass  Object  left  right  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  propertyNames  
[BugLab_Variable_Misuse]^bi = Introspector.getBeanInfo ( clazz, Object.dynamicPropertyHandlerClass ) ;^81^^^^^77^82^bi = Introspector.getBeanInfo ( clazz, Object.class ) ;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptors [RETURN_TYPE] PropertyDescriptor[]   [VARIABLES] boolean  atomic  BeanInfo  bi  IntrospectionException  ex  Class  clazz  dynamicPropertyHandlerClass  Object  left  right  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  propertyNames  
[BugLab_Variable_Misuse]^bi = Introspector.getBeanInfo ( dynamicPropertyHandlerClass ) ;^78^^^^^73^99^bi = Introspector.getBeanInfo ( clazz ) ;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptors [RETURN_TYPE] PropertyDescriptor[]   [VARIABLES] boolean  atomic  BeanInfo  bi  IntrospectionException  ex  Class  clazz  dynamicPropertyHandlerClass  Object  left  right  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  propertyNames  
[BugLab_Variable_Misuse]^bi = Introspector.getBeanInfo ( dynamicPropertyHandlerClass, Object.class ) ;^81^^^^^73^99^bi = Introspector.getBeanInfo ( clazz, Object.class ) ;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptors [RETURN_TYPE] PropertyDescriptor[]   [VARIABLES] boolean  atomic  BeanInfo  bi  IntrospectionException  ex  Class  clazz  dynamicPropertyHandlerClass  Object  left  right  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  propertyNames  
[BugLab_Variable_Misuse]^propertyDescriptors = propertyDescriptors;^92^^^^^73^99^propertyDescriptors = descriptors;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptors [RETURN_TYPE] PropertyDescriptor[]   [VARIABLES] boolean  atomic  BeanInfo  bi  IntrospectionException  ex  Class  clazz  dynamicPropertyHandlerClass  Object  left  right  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  propertyNames  
[BugLab_Variable_Misuse]^bi = Introspector.getBeanInfo ( clazz, Object.dynamicPropertyHandlerClass ) ;^81^^^^^73^99^bi = Introspector.getBeanInfo ( clazz, Object.class ) ;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptors [RETURN_TYPE] PropertyDescriptor[]   [VARIABLES] boolean  atomic  BeanInfo  bi  IntrospectionException  ex  Class  clazz  dynamicPropertyHandlerClass  Object  left  right  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  propertyNames  
[BugLab_Variable_Misuse]^System.arraycopy ( propertyDescriptors, 0, descriptors, 0, pds.length ) ;^85^^^^^73^99^System.arraycopy ( pds, 0, descriptors, 0, pds.length ) ;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptors [RETURN_TYPE] PropertyDescriptor[]   [VARIABLES] boolean  atomic  BeanInfo  bi  IntrospectionException  ex  Class  clazz  dynamicPropertyHandlerClass  Object  left  right  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  propertyNames  
[BugLab_Argument_Swapping]^System.arraycopy ( descriptors, 0, pds, 0, pds.length ) ;^85^^^^^73^99^System.arraycopy ( pds, 0, descriptors, 0, pds.length ) ;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptors [RETURN_TYPE] PropertyDescriptor[]   [VARIABLES] boolean  atomic  BeanInfo  bi  IntrospectionException  ex  Class  clazz  dynamicPropertyHandlerClass  Object  left  right  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  propertyNames  
[BugLab_Argument_Swapping]^System.arraycopy ( pds, 0, pds.length, 0, descriptors ) ;^85^^^^^73^99^System.arraycopy ( pds, 0, descriptors, 0, pds.length ) ;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptors [RETURN_TYPE] PropertyDescriptor[]   [VARIABLES] boolean  atomic  BeanInfo  bi  IntrospectionException  ex  Class  clazz  dynamicPropertyHandlerClass  Object  left  right  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  propertyNames  
[BugLab_Wrong_Literal]^System.arraycopy ( pds, 1, descriptors, 1, pds.length ) ;^85^^^^^73^99^System.arraycopy ( pds, 0, descriptors, 0, pds.length ) ;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptors [RETURN_TYPE] PropertyDescriptor[]   [VARIABLES] boolean  atomic  BeanInfo  bi  IntrospectionException  ex  Class  clazz  dynamicPropertyHandlerClass  Object  left  right  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  propertyNames  
[BugLab_Variable_Misuse]^Arrays.sort ( propertyDescriptors, new Comparator (  )  {^86^^^^^73^99^Arrays.sort ( descriptors, new Comparator (  )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptors [RETURN_TYPE] PropertyDescriptor[]   [VARIABLES] boolean  atomic  BeanInfo  bi  IntrospectionException  ex  Class  clazz  dynamicPropertyHandlerClass  Object  left  right  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  propertyNames  
[BugLab_Variable_Misuse]^System.arraycopy ( pds, 0, propertyDescriptors, 0, pds.length ) ;^85^^^^^73^99^System.arraycopy ( pds, 0, descriptors, 0, pds.length ) ;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptors [RETURN_TYPE] PropertyDescriptor[]   [VARIABLES] boolean  atomic  BeanInfo  bi  IntrospectionException  ex  Class  clazz  dynamicPropertyHandlerClass  Object  left  right  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  propertyNames  
[BugLab_Wrong_Literal]^System.arraycopy ( pds, , descriptors, , pds.length ) ;^85^^^^^73^99^System.arraycopy ( pds, 0, descriptors, 0, pds.length ) ;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptors [RETURN_TYPE] PropertyDescriptor[]   [VARIABLES] boolean  atomic  BeanInfo  bi  IntrospectionException  ex  Class  clazz  dynamicPropertyHandlerClass  Object  left  right  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  propertyNames  
[BugLab_Wrong_Literal]^System.arraycopy ( pds, -1, descriptors, -1, pds.length ) ;^85^^^^^73^99^System.arraycopy ( pds, 0, descriptors, 0, pds.length ) ;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptors [RETURN_TYPE] PropertyDescriptor[]   [VARIABLES] boolean  atomic  BeanInfo  bi  IntrospectionException  ex  Class  clazz  dynamicPropertyHandlerClass  Object  left  right  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  propertyNames  
[BugLab_Argument_Swapping]^System.arraycopy ( pds.length, 0, descriptors, 0, pds ) ;^85^^^^^73^99^System.arraycopy ( pds, 0, descriptors, 0, pds.length ) ;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptors [RETURN_TYPE] PropertyDescriptor[]   [VARIABLES] boolean  atomic  BeanInfo  bi  IntrospectionException  ex  Class  clazz  dynamicPropertyHandlerClass  Object  left  right  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  propertyNames  
[BugLab_Variable_Misuse]^return pds;^98^^^^^73^99^return propertyDescriptors;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptors [RETURN_TYPE] PropertyDescriptor[]   [VARIABLES] boolean  atomic  BeanInfo  bi  IntrospectionException  ex  Class  clazz  dynamicPropertyHandlerClass  Object  left  right  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  propertyNames  
[BugLab_Variable_Misuse]^if  ( names == null )  {^102^^^^^101^123^if  ( propertyNames == null )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Wrong_Operator]^if  ( propertyNames != null )  {^102^^^^^101^123^if  ( propertyNames == null )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Variable_Misuse]^for  ( int i = 0; i < propertyDescriptors.length; i++ )  {^105^^^^^101^123^for  ( int i = 0; i < pds.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Argument_Swapping]^for  ( pds.lengthnt i = 0; i < i; i++ )  {^105^^^^^101^123^for  ( int i = 0; i < pds.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Argument_Swapping]^for  ( int i = 0; i < pds.length.length; i++ )  {^105^^^^^101^123^for  ( int i = 0; i < pds.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= pds.length; i++ )  {^105^^^^^101^123^for  ( int i = 0; i < pds.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Variable_Misuse]^names[i] = propertyDescriptors[i].getName (  ) ;^106^^^^^101^123^names[i] = pds[i].getName (  ) ;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Variable_Misuse]^propertyNames = propertyNames;^108^^^^^101^123^propertyNames = names;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Wrong_Literal]^for  ( int i = i; i < pds.length; i++ )  {^105^^^^^101^123^for  ( int i = 0; i < pds.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Wrong_Literal]^for  ( int i = 1; i < pds.length; i++ )  {^105^^^^^101^123^for  ( int i = 0; i < pds.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Variable_Misuse]^if  ( names[i] == propertyName )  {^112^^^^^101^123^if  ( propertyNames[i] == propertyName )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Argument_Swapping]^if  ( propertyNamess[i] == propertyName )  {^112^^^^^101^123^if  ( propertyNames[i] == propertyName )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Argument_Swapping]^if  ( propertyName[i] == propertyNames )  {^112^^^^^101^123^if  ( propertyNames[i] == propertyName )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Wrong_Operator]^if  ( propertyNames[i] != propertyName )  {^112^^^^^101^123^if  ( propertyNames[i] == propertyName )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Variable_Misuse]^return pds[i];^113^^^^^101^123^return propertyDescriptors[i];^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Variable_Misuse]^for  ( int i = 0; i < names.length; i++ )  {^111^^^^^101^123^for  ( int i = 0; i < propertyNames.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Argument_Swapping]^for  ( propertyNames.lengthnt i = 0; i < i; i++ )  {^111^^^^^101^123^for  ( int i = 0; i < propertyNames.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Argument_Swapping]^for  ( propertyNament i = 0; i < is.length; i++ )  {^111^^^^^101^123^for  ( int i = 0; i < propertyNames.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Argument_Swapping]^for  ( int i = 0; i < propertyName.length; i++ )  {^111^^^^^101^123^for  ( int i = 0; i < propertyNames.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= propertyNames.length; i++ )  {^111^^^^^101^123^for  ( int i = 0; i < propertyNames.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Wrong_Literal]^for  ( int i = ; i < propertyNames.length; i++ )  {^111^^^^^101^123^for  ( int i = 0; i < propertyNames.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Variable_Misuse]^if  ( names[i].equals ( propertyName )  )  {^118^^^^^101^123^if  ( propertyNames[i].equals ( propertyName )  )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Argument_Swapping]^if  ( propertyNamess[i].equals ( propertyName )  )  {^118^^^^^101^123^if  ( propertyNames[i].equals ( propertyName )  )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Variable_Misuse]^return pds[i];^119^^^^^101^123^return propertyDescriptors[i];^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Variable_Misuse]^for  ( int i = 0; i < names.length; i++ )  {^117^^^^^101^123^for  ( int i = 0; i < propertyNames.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Argument_Swapping]^for  ( int i = 0; i < propertyNamess.length; i++ )  {^117^^^^^101^123^for  ( int i = 0; i < propertyNames.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Argument_Swapping]^for  ( propertyNames.lengthnt i = 0; i < i; i++ )  {^117^^^^^101^123^for  ( int i = 0; i < propertyNames.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Argument_Swapping]^for  ( int i = 0; i < propertyName.length; i++ )  {^117^^^^^101^123^for  ( int i = 0; i < propertyNames.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= propertyNames.length; i++ )  {^117^^^^^101^123^for  ( int i = 0; i < propertyNames.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Wrong_Literal]^for  ( int i = 1; i < propertyNames.length; i++ )  {^117^^^^^101^123^for  ( int i = 0; i < propertyNames.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Wrong_Literal]^for  ( int i = i; i < propertyNames.length; i++ )  {^117^^^^^101^123^for  ( int i = 0; i < propertyNames.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getPropertyDescriptor [RETURN_TYPE] PropertyDescriptor   String propertyName [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  String  propertyName  int  i  
[BugLab_Variable_Misuse]^return clazz;^130^^^^^129^131^return dynamicPropertyHandlerClass;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] getDynamicPropertyHandlerClass [RETURN_TYPE] Class   [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  boolean  atomic  PropertyDescriptor[]  descriptors  pds  propertyDescriptors  String[]  names  propertyNames  
[BugLab_Variable_Misuse]^buffer.append ( dynamicPropertyHandlerClass.getName (  )  ) ;^136^^^^^133^153^buffer.append ( clazz.getName (  )  ) ;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  StringBuffer  buffer  boolean  atomic  PropertyDescriptor[]  descriptors  jpds  pds  propertyDescriptors  String[]  names  propertyNames  int  i  
[BugLab_Variable_Misuse]^for  ( int i = 0; i < propertyDescriptors.length; i++ )  {^145^^^^^133^153^for  ( int i = 0; i < jpds.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  StringBuffer  buffer  boolean  atomic  PropertyDescriptor[]  descriptors  jpds  pds  propertyDescriptors  String[]  names  propertyNames  int  i  
[BugLab_Argument_Swapping]^for  ( jpdsnt i = 0; i < i.length; i++ )  {^145^^^^^133^153^for  ( int i = 0; i < jpds.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  StringBuffer  buffer  boolean  atomic  PropertyDescriptor[]  descriptors  jpds  pds  propertyDescriptors  String[]  names  propertyNames  int  i  
[BugLab_Argument_Swapping]^for  ( int i = 0; i < jpds.length.length; i++ )  {^145^^^^^133^153^for  ( int i = 0; i < jpds.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  StringBuffer  buffer  boolean  atomic  PropertyDescriptor[]  descriptors  jpds  pds  propertyDescriptors  String[]  names  propertyNames  int  i  
[BugLab_Argument_Swapping]^for  ( jpds.lengthnt i = 0; i < i; i++ )  {^145^^^^^133^153^for  ( int i = 0; i < jpds.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  StringBuffer  buffer  boolean  atomic  PropertyDescriptor[]  descriptors  jpds  pds  propertyDescriptors  String[]  names  propertyNames  int  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i == jpds.length; i++ )  {^145^^^^^133^153^for  ( int i = 0; i < jpds.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  StringBuffer  buffer  boolean  atomic  PropertyDescriptor[]  descriptors  jpds  pds  propertyDescriptors  String[]  names  propertyNames  int  i  
[BugLab_Variable_Misuse]^buffer.append ( propertyDescriptors[i].getPropertyType (  )  ) ;^147^^^^^133^153^buffer.append ( jpds[i].getPropertyType (  )  ) ;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  StringBuffer  buffer  boolean  atomic  PropertyDescriptor[]  descriptors  jpds  pds  propertyDescriptors  String[]  names  propertyNames  int  i  
[BugLab_Variable_Misuse]^buffer.append ( propertyDescriptors[i].getName (  )  ) ;^149^^^^^133^153^buffer.append ( jpds[i].getName (  )  ) ;^[CLASS] JXPathBasicBeanInfo 1  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  StringBuffer  buffer  boolean  atomic  PropertyDescriptor[]  descriptors  jpds  pds  propertyDescriptors  String[]  names  propertyNames  int  i  
[BugLab_Wrong_Literal]^for  ( int i = 1; i < jpds.length; i++ )  {^145^^^^^133^153^for  ( int i = 0; i < jpds.length; i++ )  {^[CLASS] JXPathBasicBeanInfo 1  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] Class  clazz  dynamicPropertyHandlerClass  StringBuffer  buffer  boolean  atomic  PropertyDescriptor[]  descriptors  jpds  pds  propertyDescriptors  String[]  names  propertyNames  int  i  
