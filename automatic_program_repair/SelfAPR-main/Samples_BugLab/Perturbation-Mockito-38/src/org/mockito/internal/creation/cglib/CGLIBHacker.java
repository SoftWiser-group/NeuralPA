[BugLab_Variable_Misuse]^if  ( createInfoField.get ( createInfo )  == null )  {^20^^^^^13^26^if  ( namingPolicyField.get ( createInfo )  == null )  {^[CLASS] CGLIBHacker  [METHOD] setMockitoNamingPolicy [RETURN_TYPE] void   MethodProxy methodProxy [VARIABLES] Field  createInfoField  namingPolicyField  boolean  MethodProxy  methodProxy  Object  createInfo  Exception  e  
[BugLab_Argument_Swapping]^if  ( createInfo.get ( namingPolicyField )  == null )  {^20^^^^^13^26^if  ( namingPolicyField.get ( createInfo )  == null )  {^[CLASS] CGLIBHacker  [METHOD] setMockitoNamingPolicy [RETURN_TYPE] void   MethodProxy methodProxy [VARIABLES] Field  createInfoField  namingPolicyField  boolean  MethodProxy  methodProxy  Object  createInfo  Exception  e  
[BugLab_Wrong_Operator]^if  ( namingPolicyField.get ( createInfo )  != null )  {^20^^^^^13^26^if  ( namingPolicyField.get ( createInfo )  == null )  {^[CLASS] CGLIBHacker  [METHOD] setMockitoNamingPolicy [RETURN_TYPE] void   MethodProxy methodProxy [VARIABLES] Field  createInfoField  namingPolicyField  boolean  MethodProxy  methodProxy  Object  createInfo  Exception  e  
[BugLab_Argument_Swapping]^Object createInfo = methodProxy.get ( createInfoField ) ;^17^^^^^13^26^Object createInfo = createInfoField.get ( methodProxy ) ;^[CLASS] CGLIBHacker  [METHOD] setMockitoNamingPolicy [RETURN_TYPE] void   MethodProxy methodProxy [VARIABLES] Field  createInfoField  namingPolicyField  boolean  MethodProxy  methodProxy  Object  createInfo  Exception  e  
[BugLab_Wrong_Literal]^createInfoField.setAccessible ( false ) ;^16^^^^^13^26^createInfoField.setAccessible ( true ) ;^[CLASS] CGLIBHacker  [METHOD] setMockitoNamingPolicy [RETURN_TYPE] void   MethodProxy methodProxy [VARIABLES] Field  createInfoField  namingPolicyField  boolean  MethodProxy  methodProxy  Object  createInfo  Exception  e  
[BugLab_Variable_Misuse]^Object createInfo = namingPolicyField.get ( methodProxy ) ;^17^^^^^13^26^Object createInfo = createInfoField.get ( methodProxy ) ;^[CLASS] CGLIBHacker  [METHOD] setMockitoNamingPolicy [RETURN_TYPE] void   MethodProxy methodProxy [VARIABLES] Field  createInfoField  namingPolicyField  boolean  MethodProxy  methodProxy  Object  createInfo  Exception  e  
[BugLab_Wrong_Literal]^namingPolicyField.setAccessible ( false ) ;^19^^^^^13^26^namingPolicyField.setAccessible ( true ) ;^[CLASS] CGLIBHacker  [METHOD] setMockitoNamingPolicy [RETURN_TYPE] void   MethodProxy methodProxy [VARIABLES] Field  createInfoField  namingPolicyField  boolean  MethodProxy  methodProxy  Object  createInfo  Exception  e  
[BugLab_Variable_Misuse]^while  ( cglibMethodProxyClass != MethodProxy.cglibMethodProxyClass )  {^33^^^^^29^37^while  ( cglibMethodProxyClass != MethodProxy.class )  {^[CLASS] CGLIBHacker  [METHOD] reflectOnCreateInfo [RETURN_TYPE] Field   MethodProxy methodProxy [VARIABLES] boolean  MethodProxy  methodProxy  Class  cglibMethodProxyClass  
[BugLab_Wrong_Operator]^while  ( cglibMethodProxyClass == MethodProxy.class )  {^33^^^^^29^37^while  ( cglibMethodProxyClass != MethodProxy.class )  {^[CLASS] CGLIBHacker  [METHOD] reflectOnCreateInfo [RETURN_TYPE] Field   MethodProxy methodProxy [VARIABLES] boolean  MethodProxy  methodProxy  Class  cglibMethodProxyClass  
