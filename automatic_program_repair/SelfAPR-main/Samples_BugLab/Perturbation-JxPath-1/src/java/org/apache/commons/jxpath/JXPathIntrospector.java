[BugLab_Wrong_Literal]^byClass.put ( beanClass, new JXPathBasicBeanInfo ( beanClass, false )  ) ;^67^^^^^66^68^byClass.put ( beanClass, new JXPathBasicBeanInfo ( beanClass, true )  ) ;^[CLASS] JXPathIntrospector  [METHOD] registerAtomicClass [RETURN_TYPE] void   Class beanClass [VARIABLES] HashMap  byClass  byInterface  Class  beanClass  boolean  
[BugLab_Argument_Swapping]^new JXPathBasicBeanInfo ( dynamicPropertyHandlerClass, beanClass ) ;^79^^^^^74^86^new JXPathBasicBeanInfo ( beanClass, dynamicPropertyHandlerClass ) ;^[CLASS] JXPathIntrospector  [METHOD] registerDynamicClass [RETURN_TYPE] void   Class beanClass Class dynamicPropertyHandlerClass [VARIABLES] Class  beanClass  dynamicPropertyHandlerClass  JXPathBasicBeanInfo  bi  boolean  HashMap  byClass  byInterface  
[BugLab_Variable_Misuse]^JXPathBasicBeanInfo bi = new JXPathBasicBeanInfo ( dynamicPropertyHandlerClass, dynamicPropertyHandlerClass ) ;^78^79^^^^74^86^JXPathBasicBeanInfo bi = new JXPathBasicBeanInfo ( beanClass, dynamicPropertyHandlerClass ) ;^[CLASS] JXPathIntrospector  [METHOD] registerDynamicClass [RETURN_TYPE] void   Class beanClass Class dynamicPropertyHandlerClass [VARIABLES] Class  beanClass  dynamicPropertyHandlerClass  JXPathBasicBeanInfo  bi  boolean  HashMap  byClass  byInterface  
[BugLab_Variable_Misuse]^JXPathBasicBeanInfo bi = new JXPathBasicBeanInfo ( beanClass, beanClass ) ;^78^79^^^^74^86^JXPathBasicBeanInfo bi = new JXPathBasicBeanInfo ( beanClass, dynamicPropertyHandlerClass ) ;^[CLASS] JXPathIntrospector  [METHOD] registerDynamicClass [RETURN_TYPE] void   Class beanClass Class dynamicPropertyHandlerClass [VARIABLES] Class  beanClass  dynamicPropertyHandlerClass  JXPathBasicBeanInfo  bi  boolean  HashMap  byClass  byInterface  
[BugLab_Argument_Swapping]^JXPathBasicBeanInfo bi = new JXPathBasicBeanInfo ( dynamicPropertyHandlerClass, beanClass ) ;^78^79^^^^74^86^JXPathBasicBeanInfo bi = new JXPathBasicBeanInfo ( beanClass, dynamicPropertyHandlerClass ) ;^[CLASS] JXPathIntrospector  [METHOD] registerDynamicClass [RETURN_TYPE] void   Class beanClass Class dynamicPropertyHandlerClass [VARIABLES] Class  beanClass  dynamicPropertyHandlerClass  JXPathBasicBeanInfo  bi  boolean  HashMap  byClass  byInterface  
[BugLab_Variable_Misuse]^if  ( dynamicPropertyHandlerClass.isInterface (  )  )  {^80^^^^^74^86^if  ( beanClass.isInterface (  )  )  {^[CLASS] JXPathIntrospector  [METHOD] registerDynamicClass [RETURN_TYPE] void   Class beanClass Class dynamicPropertyHandlerClass [VARIABLES] Class  beanClass  dynamicPropertyHandlerClass  JXPathBasicBeanInfo  bi  boolean  HashMap  byClass  byInterface  
[BugLab_Variable_Misuse]^byInterface.put ( dynamicPropertyHandlerClass, bi ) ;^81^^^^^74^86^byInterface.put ( beanClass, bi ) ;^[CLASS] JXPathIntrospector  [METHOD] registerDynamicClass [RETURN_TYPE] void   Class beanClass Class dynamicPropertyHandlerClass [VARIABLES] Class  beanClass  dynamicPropertyHandlerClass  JXPathBasicBeanInfo  bi  boolean  HashMap  byClass  byInterface  
[BugLab_Argument_Swapping]^byInterface.put ( bi, beanClass ) ;^81^^^^^74^86^byInterface.put ( beanClass, bi ) ;^[CLASS] JXPathIntrospector  [METHOD] registerDynamicClass [RETURN_TYPE] void   Class beanClass Class dynamicPropertyHandlerClass [VARIABLES] Class  beanClass  dynamicPropertyHandlerClass  JXPathBasicBeanInfo  bi  boolean  HashMap  byClass  byInterface  
[BugLab_Variable_Misuse]^byClass.put ( dynamicPropertyHandlerClass, bi ) ;^84^^^^^74^86^byClass.put ( beanClass, bi ) ;^[CLASS] JXPathIntrospector  [METHOD] registerDynamicClass [RETURN_TYPE] void   Class beanClass Class dynamicPropertyHandlerClass [VARIABLES] Class  beanClass  dynamicPropertyHandlerClass  JXPathBasicBeanInfo  bi  boolean  HashMap  byClass  byInterface  
[BugLab_Argument_Swapping]^byClass.put ( bi, beanClass ) ;^84^^^^^74^86^byClass.put ( beanClass, bi ) ;^[CLASS] JXPathIntrospector  [METHOD] registerDynamicClass [RETURN_TYPE] void   Class beanClass Class dynamicPropertyHandlerClass [VARIABLES] Class  beanClass  dynamicPropertyHandlerClass  JXPathBasicBeanInfo  bi  boolean  HashMap  byClass  byInterface  
[BugLab_Variable_Misuse]^JXPathBeanInfo beanInfo =  ( JXPathBeanInfo )  byInterface.get ( beanClass ) ;^102^^^^^101^114^JXPathBeanInfo beanInfo =  ( JXPathBeanInfo )  byClass.get ( beanClass ) ;^[CLASS] JXPathIntrospector  [METHOD] getBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  boolean  HashMap  byClass  byInterface  
[BugLab_Argument_Swapping]^JXPathBeanInfo beanInfo =  ( JXPathBeanInfo )  beanClass.get ( byClass ) ;^102^^^^^101^114^JXPathBeanInfo beanInfo =  ( JXPathBeanInfo )  byClass.get ( beanClass ) ;^[CLASS] JXPathIntrospector  [METHOD] getBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  boolean  HashMap  byClass  byInterface  
[BugLab_Wrong_Operator]^if  ( beanInfo != null )  {^103^^^^^101^114^if  ( beanInfo == null )  {^[CLASS] JXPathIntrospector  [METHOD] getBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  boolean  HashMap  byClass  byInterface  
[BugLab_Wrong_Operator]^if  ( beanInfo != null )  {^105^^^^^101^114^if  ( beanInfo == null )  {^[CLASS] JXPathIntrospector  [METHOD] getBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  boolean  HashMap  byClass  byInterface  
[BugLab_Wrong_Operator]^if  ( beanInfo != null )  {^107^^^^^101^114^if  ( beanInfo == null )  {^[CLASS] JXPathIntrospector  [METHOD] getBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  boolean  HashMap  byClass  byInterface  
[BugLab_Argument_Swapping]^byClass.put ( beanInfo, beanClass ) ;^111^^^^^101^114^byClass.put ( beanClass, beanInfo ) ;^[CLASS] JXPathIntrospector  [METHOD] getBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  boolean  HashMap  byClass  byInterface  
[BugLab_Variable_Misuse]^if  ( sup.isInterface (  )  )  {^122^^^^^120^148^if  ( beanClass.isInterface (  )  )  {^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Wrong_Operator]^if  ( beanInfo != null || beanInfo.isDynamic (  )  )  {^124^^^^^120^148^if  ( beanInfo != null && beanInfo.isDynamic (  )  )  {^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Wrong_Operator]^if  ( beanInfo == null && beanInfo.isDynamic (  )  )  {^124^^^^^120^148^if  ( beanInfo != null && beanInfo.isDynamic (  )  )  {^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Variable_Misuse]^beanInfo =  ( JXPathBeanInfo )  byInterface.get ( sup ) ;^123^^^^^120^148^beanInfo =  ( JXPathBeanInfo )  byInterface.get ( beanClass ) ;^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Variable_Misuse]^beanInfo =  ( JXPathBeanInfo )  byClass.get ( beanClass ) ;^123^^^^^120^148^beanInfo =  ( JXPathBeanInfo )  byInterface.get ( beanClass ) ;^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Argument_Swapping]^beanInfo =  ( JXPathBeanInfo )  beanClass.get ( byInterface ) ;^123^^^^^120^148^beanInfo =  ( JXPathBeanInfo )  byInterface.get ( beanClass ) ;^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Variable_Misuse]^Class interfaces[] = sup.getInterfaces (  ) ;^129^^^^^120^148^Class interfaces[] = beanClass.getInterfaces (  ) ;^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Wrong_Operator]^if  ( interfaces == null )  {^130^^^^^120^148^if  ( interfaces != null )  {^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Wrong_Operator]^if  ( beanInfo != null || beanInfo.isDynamic (  )  )  {^133^^^^^120^148^if  ( beanInfo != null && beanInfo.isDynamic (  )  )  {^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Wrong_Operator]^if  ( beanInfo == null && beanInfo.isDynamic (  )  )  {^133^^^^^120^148^if  ( beanInfo != null && beanInfo.isDynamic (  )  )  {^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Argument_Swapping]^for  ( interfaces.lengthnt i = 0; i < i; i++ )  {^131^^^^^120^148^for  ( int i = 0; i < interfaces.length; i++ )  {^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Argument_Swapping]^for  ( int i = 0; i < interfaces.length.length; i++ )  {^131^^^^^120^148^for  ( int i = 0; i < interfaces.length; i++ )  {^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i > interfaces.length; i++ )  {^131^^^^^120^148^for  ( int i = 0; i < interfaces.length; i++ )  {^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Wrong_Literal]^for  ( int i = i; i < interfaces.length; i++ )  {^131^^^^^120^148^for  ( int i = 0; i < interfaces.length; i++ )  {^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Argument_Swapping]^beanInfo = findDynamicBeanInfo ( i[i] ) ;^132^^^^^120^148^beanInfo = findDynamicBeanInfo ( interfaces[i] ) ;^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Argument_Swapping]^beanInfo = findDynamicBeanInfo ( interfacesnterfaces[i] ) ;^132^^^^^120^148^beanInfo = findDynamicBeanInfo ( interfaces[i] ) ;^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Wrong_Literal]^for  ( int i = 1; i < interfaces.length; i++ )  {^131^^^^^120^148^for  ( int i = 0; i < interfaces.length; i++ )  {^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= interfaces.length; i++ )  {^131^^^^^120^148^for  ( int i = 0; i < interfaces.length; i++ )  {^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Variable_Misuse]^Class sup = sup.getSuperclass (  ) ;^139^^^^^120^148^Class sup = beanClass.getSuperclass (  ) ;^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Wrong_Operator]^if  ( sup == null )  {^140^^^^^120^148^if  ( sup != null )  {^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Wrong_Operator]^if  ( beanInfo != null || beanInfo.isDynamic (  )  )  {^142^^^^^120^148^if  ( beanInfo != null && beanInfo.isDynamic (  )  )  {^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Wrong_Operator]^if  ( beanInfo == null && beanInfo.isDynamic (  )  )  {^142^^^^^120^148^if  ( beanInfo != null && beanInfo.isDynamic (  )  )  {^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Variable_Misuse]^beanInfo =  ( JXPathBeanInfo )  byClass.get ( beanClass ) ;^141^^^^^120^148^beanInfo =  ( JXPathBeanInfo )  byClass.get ( sup ) ;^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Variable_Misuse]^beanInfo =  ( JXPathBeanInfo )  byInterface.get ( sup ) ;^141^^^^^120^148^beanInfo =  ( JXPathBeanInfo )  byClass.get ( sup ) ;^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Argument_Swapping]^beanInfo =  ( JXPathBeanInfo )  sup.get ( byClass ) ;^141^^^^^120^148^beanInfo =  ( JXPathBeanInfo )  byClass.get ( sup ) ;^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Variable_Misuse]^return findDynamicBeanInfo ( beanClass ) ;^145^^^^^120^148^return findDynamicBeanInfo ( sup ) ;^[CLASS] JXPathIntrospector  [METHOD] findDynamicBeanInfo [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] JXPathBeanInfo  beanInfo  Class  beanClass  sup  Class[]  interfaces  boolean  HashMap  byClass  byInterface  int  i  
[BugLab_Wrong_Operator]^String name = beanClass.getName (  &  )  + "XBeanInfo";^151^^^^^150^170^String name = beanClass.getName (  )  + "XBeanInfo";^[CLASS] JXPathIntrospector  [METHOD] findInformant [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] Class  beanClass  String  name  boolean  HashMap  byClass  byInterface  Exception  ex  
[BugLab_Argument_Swapping]^return  ( JXPathBeanInfo )  instantiate ( name, beanClass ) ;^153^^^^^150^170^return  ( JXPathBeanInfo )  instantiate ( beanClass, name ) ;^[CLASS] JXPathIntrospector  [METHOD] findInformant [RETURN_TYPE] JXPathBeanInfo   Class beanClass [VARIABLES] Class  beanClass  String  name  boolean  HashMap  byClass  byInterface  Exception  ex  
[BugLab_Variable_Misuse]^ClassLoader cl = cls.getClassLoader (  ) ;^182^^^^^177^196^ClassLoader cl = sibling.getClassLoader (  ) ;^[CLASS] JXPathIntrospector  [METHOD] instantiate [RETURN_TYPE] Object   Class sibling String className [VARIABLES] ClassLoader  cl  Class  cls  sibling  String  className  boolean  HashMap  byClass  byInterface  Exception  ex  
[BugLab_Wrong_Operator]^if  ( cl == null )  {^183^^^^^177^196^if  ( cl != null )  {^[CLASS] JXPathIntrospector  [METHOD] instantiate [RETURN_TYPE] Object   Class sibling String className [VARIABLES] ClassLoader  cl  Class  cls  sibling  String  className  boolean  HashMap  byClass  byInterface  Exception  ex  
[BugLab_Variable_Misuse]^return sibling.newInstance (  ) ;^186^^^^^177^196^return cls.newInstance (  ) ;^[CLASS] JXPathIntrospector  [METHOD] instantiate [RETURN_TYPE] Object   Class sibling String className [VARIABLES] ClassLoader  cl  Class  cls  sibling  String  className  boolean  HashMap  byClass  byInterface  Exception  ex  
[BugLab_Argument_Swapping]^Class classNames = cl.loadClass ( cl ) ;^185^^^^^177^196^Class cls = cl.loadClass ( className ) ;^[CLASS] JXPathIntrospector  [METHOD] instantiate [RETURN_TYPE] Object   Class sibling String className [VARIABLES] ClassLoader  cl  Class  cls  sibling  String  className  boolean  HashMap  byClass  byInterface  Exception  ex  
[BugLab_Variable_Misuse]^return sibling.newInstance (  ) ;^195^^^^^177^196^return cls.newInstance (  ) ;^[CLASS] JXPathIntrospector  [METHOD] instantiate [RETURN_TYPE] Object   Class sibling String className [VARIABLES] ClassLoader  cl  Class  cls  sibling  String  className  boolean  HashMap  byClass  byInterface  Exception  ex  
