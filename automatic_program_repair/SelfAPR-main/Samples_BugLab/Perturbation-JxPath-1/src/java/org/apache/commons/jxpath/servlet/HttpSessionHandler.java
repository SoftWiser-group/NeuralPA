[BugLab_Wrong_Operator]^if  ( session == null )  {^40^^^^^35^46^if  ( session != null )  {^[CLASS] HttpSessionHandler  [METHOD] collectPropertyNames [RETURN_TYPE] void   HashSet set Object bean [VARIABLES] Enumeration  e  boolean  HttpSessionAndServletContext  handle  HttpSession  session  Object  bean  HashSet  set  
[BugLab_Wrong_Operator]^if  ( session == null )  {^52^^^^^48^59^if  ( session != null )  {^[CLASS] HttpSessionHandler  [METHOD] getProperty [RETURN_TYPE] Object   Object bean String property [VARIABLES] boolean  HttpSessionAndServletContext  handle  HttpSession  session  Object  bean  object  String  property  
[BugLab_Variable_Misuse]^if  ( bean != null )  {^54^^^^^48^59^if  ( object != null )  {^[CLASS] HttpSessionHandler  [METHOD] getProperty [RETURN_TYPE] Object   Object bean String property [VARIABLES] boolean  HttpSessionAndServletContext  handle  HttpSession  session  Object  bean  object  String  property  
[BugLab_Wrong_Operator]^if  ( object == null )  {^54^^^^^48^59^if  ( object != null )  {^[CLASS] HttpSessionHandler  [METHOD] getProperty [RETURN_TYPE] Object   Object bean String property [VARIABLES] boolean  HttpSessionAndServletContext  handle  HttpSession  session  Object  bean  object  String  property  
[BugLab_Variable_Misuse]^return bean;^55^^^^^48^59^return object;^[CLASS] HttpSessionHandler  [METHOD] getProperty [RETURN_TYPE] Object   Object bean String property [VARIABLES] boolean  HttpSessionAndServletContext  handle  HttpSession  session  Object  bean  object  String  property  
[BugLab_Argument_Swapping]^Object object = property.getAttribute ( session ) ;^53^^^^^48^59^Object object = session.getAttribute ( property ) ;^[CLASS] HttpSessionHandler  [METHOD] getProperty [RETURN_TYPE] Object   Object bean String property [VARIABLES] boolean  HttpSessionAndServletContext  handle  HttpSession  session  Object  bean  object  String  property  
[BugLab_Wrong_Operator]^if  ( session == null )  {^65^^^^^61^72^if  ( session != null )  {^[CLASS] HttpSessionHandler  [METHOD] setProperty [RETURN_TYPE] void   Object bean String property Object value [VARIABLES] boolean  HttpSessionAndServletContext  handle  HttpSession  session  Object  bean  value  String  property  
[BugLab_Wrong_Operator]^throw new JXPathException ( "Cannot set session attribute: "  >>  "there is no session" ) ;^69^70^^^^61^72^throw new JXPathException ( "Cannot set session attribute: " + "there is no session" ) ;^[CLASS] HttpSessionHandler  [METHOD] setProperty [RETURN_TYPE] void   Object bean String property Object value [VARIABLES] boolean  HttpSessionAndServletContext  handle  HttpSession  session  Object  bean  value  String  property  
[BugLab_Wrong_Operator]^throw new JXPathException ( "Cannot set session attribute: "   instanceof   "there is no session" ) ;^69^70^^^^61^72^throw new JXPathException ( "Cannot set session attribute: " + "there is no session" ) ;^[CLASS] HttpSessionHandler  [METHOD] setProperty [RETURN_TYPE] void   Object bean String property Object value [VARIABLES] boolean  HttpSessionAndServletContext  handle  HttpSession  session  Object  bean  value  String  property  
[BugLab_Variable_Misuse]^session.setAttribute ( property, bean ) ;^66^^^^^61^72^session.setAttribute ( property, value ) ;^[CLASS] HttpSessionHandler  [METHOD] setProperty [RETURN_TYPE] void   Object bean String property Object value [VARIABLES] boolean  HttpSessionAndServletContext  handle  HttpSession  session  Object  bean  value  String  property  
[BugLab_Argument_Swapping]^session.setAttribute ( value, property ) ;^66^^^^^61^72^session.setAttribute ( property, value ) ;^[CLASS] HttpSessionHandler  [METHOD] setProperty [RETURN_TYPE] void   Object bean String property Object value [VARIABLES] boolean  HttpSessionAndServletContext  handle  HttpSession  session  Object  bean  value  String  property  
[BugLab_Wrong_Operator]^throw new JXPathException ( "Cannot set session attribute: "  <<  "there is no session" ) ;^69^70^^^^61^72^throw new JXPathException ( "Cannot set session attribute: " + "there is no session" ) ;^[CLASS] HttpSessionHandler  [METHOD] setProperty [RETURN_TYPE] void   Object bean String property Object value [VARIABLES] boolean  HttpSessionAndServletContext  handle  HttpSession  session  Object  bean  value  String  property  
