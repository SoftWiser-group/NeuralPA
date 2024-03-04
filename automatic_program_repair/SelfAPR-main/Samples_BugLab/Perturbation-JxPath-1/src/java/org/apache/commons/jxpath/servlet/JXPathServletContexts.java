[BugLab_Wrong_Operator]^if  ( context != null )  {^97^^^^^94^110^if  ( context == null )  {^[CLASS] JXPathServletContexts  [METHOD] getPageContext [RETURN_TYPE] JXPathContext   PageContext pageContext [VARIABLES] JXPathContextFactory  factory  PageContext  pageContext  boolean  JXPathContext  context  parentContext  
[BugLab_Variable_Misuse]^context = factory.newContext ( context, pageContext ) ;^102^^^^^94^110^context = factory.newContext ( parentContext, pageContext ) ;^[CLASS] JXPathServletContexts  [METHOD] getPageContext [RETURN_TYPE] JXPathContext   PageContext pageContext [VARIABLES] JXPathContextFactory  factory  PageContext  pageContext  boolean  JXPathContext  context  parentContext  
[BugLab_Argument_Swapping]^context = parentContext.newContext ( factory, pageContext ) ;^102^^^^^94^110^context = factory.newContext ( parentContext, pageContext ) ;^[CLASS] JXPathServletContexts  [METHOD] getPageContext [RETURN_TYPE] JXPathContext   PageContext pageContext [VARIABLES] JXPathContextFactory  factory  PageContext  pageContext  boolean  JXPathContext  context  parentContext  
[BugLab_Argument_Swapping]^context = factory.newContext ( pageContext, parentContext ) ;^102^^^^^94^110^context = factory.newContext ( parentContext, pageContext ) ;^[CLASS] JXPathServletContexts  [METHOD] getPageContext [RETURN_TYPE] JXPathContext   PageContext pageContext [VARIABLES] JXPathContextFactory  factory  PageContext  pageContext  boolean  JXPathContext  context  parentContext  
[BugLab_Argument_Swapping]^new KeywordVariables ( pageContext, new PageScopeContext ( Constants.PAGE_SCOPE )  )  ) ;^104^105^106^^^94^110^new KeywordVariables ( Constants.PAGE_SCOPE, new PageScopeContext ( pageContext )  )  ) ;^[CLASS] JXPathServletContexts  [METHOD] getPageContext [RETURN_TYPE] JXPathContext   PageContext pageContext [VARIABLES] JXPathContextFactory  factory  PageContext  pageContext  boolean  JXPathContext  context  parentContext  
[BugLab_Argument_Swapping]^context = pageContext.newContext ( parentContext, factory ) ;^102^^^^^94^110^context = factory.newContext ( parentContext, pageContext ) ;^[CLASS] JXPathServletContexts  [METHOD] getPageContext [RETURN_TYPE] JXPathContext   PageContext pageContext [VARIABLES] JXPathContextFactory  factory  PageContext  pageContext  boolean  JXPathContext  context  parentContext  
[BugLab_Argument_Swapping]^context.setVariables ( new KeywordVariables ( pageContext, new PageScopeContext ( Constants.PAGE_SCOPE )  )  ) ;^103^104^105^106^^94^110^context.setVariables ( new KeywordVariables ( Constants.PAGE_SCOPE, new PageScopeContext ( pageContext )  )  ) ;^[CLASS] JXPathServletContexts  [METHOD] getPageContext [RETURN_TYPE] JXPathContext   PageContext pageContext [VARIABLES] JXPathContextFactory  factory  PageContext  pageContext  boolean  JXPathContext  context  parentContext  
[BugLab_Variable_Misuse]^pageContext.setAttribute ( Constants.JXPATH_CONTEXT, parentContext ) ;^107^^^^^94^110^pageContext.setAttribute ( Constants.JXPATH_CONTEXT, context ) ;^[CLASS] JXPathServletContexts  [METHOD] getPageContext [RETURN_TYPE] JXPathContext   PageContext pageContext [VARIABLES] JXPathContextFactory  factory  PageContext  pageContext  boolean  JXPathContext  context  parentContext  
[BugLab_Variable_Misuse]^return parentContext;^109^^^^^94^110^return context;^[CLASS] JXPathServletContexts  [METHOD] getPageContext [RETURN_TYPE] JXPathContext   PageContext pageContext [VARIABLES] JXPathContextFactory  factory  PageContext  pageContext  boolean  JXPathContext  context  parentContext  
[BugLab_Variable_Misuse]^if  ( parentContext != null )  {^125^^^^^110^140^if  ( context != null )  {^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Wrong_Operator]^if  ( context == null )  {^125^^^^^110^140^if  ( context != null )  {^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Argument_Swapping]^if  ( request.getServletRequest (  )  == handle )  {^128^^^^^113^143^if  ( handle.getServletRequest (  )  == request )  {^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Wrong_Operator]^if  ( handle.getServletRequest (  )  != request )  {^128^^^^^113^143^if  ( handle.getServletRequest (  )  == request )  {^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Variable_Misuse]^return parentContext;^129^^^^^114^144^return context;^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Variable_Misuse]^ServletRequestAndContext handle = ( ServletRequestAndContext )  parentContext.getContextBean (  ) ;^126^127^^^^111^141^ServletRequestAndContext handle = ( ServletRequestAndContext )  context.getContextBean (  ) ;^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Variable_Misuse]^( ServletRequestAndContext )  parentContext.getContextBean (  ) ;^127^^^^^112^142^( ServletRequestAndContext )  context.getContextBean (  ) ;^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Wrong_Operator]^if  ( request  &&  HttpServletRequest )  {^134^^^^^119^149^if  ( request instanceof HttpServletRequest )  {^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Wrong_Operator]^if  ( session == null )  {^137^^^^^122^152^if  ( session != null )  {^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Argument_Swapping]^parentContext = getSessionContext ( servletContext, session ) ;^138^^^^^123^153^parentContext = getSessionContext ( session, servletContext ) ;^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Wrong_Literal]^HttpSession session = (  ( HttpServletRequest )  request ) .getSession ( true ) ;^135^136^^^^120^150^HttpSession session = (  ( HttpServletRequest )  request ) .getSession ( false ) ;^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Wrong_Literal]^(  ( HttpServletRequest )  request ) .getSession ( true ) ;^136^^^^^121^151^(  ( HttpServletRequest )  request ) .getSession ( false ) ;^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Argument_Swapping]^new ServletRequestAndContext ( servletContext, request ) ;^145^^^^^130^160^new ServletRequestAndContext ( request, servletContext ) ;^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Argument_Swapping]^ServletRequestAndContext handle = new ServletRequestAndContext ( servletContext, request ) ;^144^145^^^^129^159^ServletRequestAndContext handle = new ServletRequestAndContext ( request, servletContext ) ;^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Argument_Swapping]^context = parentContext.newContext ( factory, handle ) ;^146^^^^^131^161^context = factory.newContext ( parentContext, handle ) ;^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Argument_Swapping]^context = handle.newContext ( parentContext, factory ) ;^146^^^^^131^161^context = factory.newContext ( parentContext, handle ) ;^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Variable_Misuse]^context = factory.newContext ( context, handle ) ;^146^^^^^131^161^context = factory.newContext ( parentContext, handle ) ;^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Argument_Swapping]^new KeywordVariables ( handle, Constants.REQUEST_SCOPE )  ) ;^148^^^^^133^163^new KeywordVariables ( Constants.REQUEST_SCOPE, handle )  ) ;^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Argument_Swapping]^context.setVariables ( new KeywordVariables ( handle, Constants.REQUEST_SCOPE )  ) ;^147^148^^^^132^162^context.setVariables ( new KeywordVariables ( Constants.REQUEST_SCOPE, handle )  ) ;^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Variable_Misuse]^request.setAttribute ( Constants.JXPATH_CONTEXT, parentContext ) ;^149^^^^^134^164^request.setAttribute ( Constants.JXPATH_CONTEXT, context ) ;^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Variable_Misuse]^return parentContext;^150^^^^^135^165^return context;^[CLASS] JXPathServletContexts  [METHOD] getRequestContext [RETURN_TYPE] JXPathContext   ServletRequest request ServletContext servletContext [VARIABLES] ServletRequest  request  boolean  JXPathContext  context  parentContext  HttpSession  session  ServletContext  servletContext  JXPathContextFactory  factory  ServletRequestAndContext  handle  
[BugLab_Wrong_Operator]^if  ( context != null )  {^163^^^^^157^173^if  ( context == null )  {^[CLASS] JXPathServletContexts  [METHOD] getSessionContext [RETURN_TYPE] JXPathContext   HttpSession session ServletContext servletContext [VARIABLES] ServletContext  servletContext  JXPathContextFactory  factory  boolean  JXPathContext  context  parentContext  HttpSession  session  HttpSessionAndServletContext  handle  
[BugLab_Variable_Misuse]^context = factory.newContext ( context, handle ) ;^167^^^^^157^173^context = factory.newContext ( parentContext, handle ) ;^[CLASS] JXPathServletContexts  [METHOD] getSessionContext [RETURN_TYPE] JXPathContext   HttpSession session ServletContext servletContext [VARIABLES] ServletContext  servletContext  JXPathContextFactory  factory  boolean  JXPathContext  context  parentContext  HttpSession  session  HttpSessionAndServletContext  handle  
[BugLab_Argument_Swapping]^context = parentContext.newContext ( factory, handle ) ;^167^^^^^157^173^context = factory.newContext ( parentContext, handle ) ;^[CLASS] JXPathServletContexts  [METHOD] getSessionContext [RETURN_TYPE] JXPathContext   HttpSession session ServletContext servletContext [VARIABLES] ServletContext  servletContext  JXPathContextFactory  factory  boolean  JXPathContext  context  parentContext  HttpSession  session  HttpSessionAndServletContext  handle  
[BugLab_Argument_Swapping]^context = handle.newContext ( parentContext, factory ) ;^167^^^^^157^173^context = factory.newContext ( parentContext, handle ) ;^[CLASS] JXPathServletContexts  [METHOD] getSessionContext [RETURN_TYPE] JXPathContext   HttpSession session ServletContext servletContext [VARIABLES] ServletContext  servletContext  JXPathContextFactory  factory  boolean  JXPathContext  context  parentContext  HttpSession  session  HttpSessionAndServletContext  handle  
[BugLab_Argument_Swapping]^new HttpSessionAndServletContext ( servletContext, session ) ;^166^^^^^157^173^new HttpSessionAndServletContext ( session, servletContext ) ;^[CLASS] JXPathServletContexts  [METHOD] getSessionContext [RETURN_TYPE] JXPathContext   HttpSession session ServletContext servletContext [VARIABLES] ServletContext  servletContext  JXPathContextFactory  factory  boolean  JXPathContext  context  parentContext  HttpSession  session  HttpSessionAndServletContext  handle  
[BugLab_Argument_Swapping]^new KeywordVariables ( handle, Constants.SESSION_SCOPE )  ) ;^169^^^^^157^173^new KeywordVariables ( Constants.SESSION_SCOPE, handle )  ) ;^[CLASS] JXPathServletContexts  [METHOD] getSessionContext [RETURN_TYPE] JXPathContext   HttpSession session ServletContext servletContext [VARIABLES] ServletContext  servletContext  JXPathContextFactory  factory  boolean  JXPathContext  context  parentContext  HttpSession  session  HttpSessionAndServletContext  handle  
[BugLab_Argument_Swapping]^HttpSessionAndServletContext handle = new HttpSessionAndServletContext ( servletContext, session ) ;^165^166^^^^157^173^HttpSessionAndServletContext handle = new HttpSessionAndServletContext ( session, servletContext ) ;^[CLASS] JXPathServletContexts  [METHOD] getSessionContext [RETURN_TYPE] JXPathContext   HttpSession session ServletContext servletContext [VARIABLES] ServletContext  servletContext  JXPathContextFactory  factory  boolean  JXPathContext  context  parentContext  HttpSession  session  HttpSessionAndServletContext  handle  
[BugLab_Argument_Swapping]^context = factory.newContext ( handle, parentContext ) ;^167^^^^^157^173^context = factory.newContext ( parentContext, handle ) ;^[CLASS] JXPathServletContexts  [METHOD] getSessionContext [RETURN_TYPE] JXPathContext   HttpSession session ServletContext servletContext [VARIABLES] ServletContext  servletContext  JXPathContextFactory  factory  boolean  JXPathContext  context  parentContext  HttpSession  session  HttpSessionAndServletContext  handle  
[BugLab_Argument_Swapping]^context.setVariables ( new KeywordVariables ( handle, Constants.SESSION_SCOPE )  ) ;^168^169^^^^157^173^context.setVariables ( new KeywordVariables ( Constants.SESSION_SCOPE, handle )  ) ;^[CLASS] JXPathServletContexts  [METHOD] getSessionContext [RETURN_TYPE] JXPathContext   HttpSession session ServletContext servletContext [VARIABLES] ServletContext  servletContext  JXPathContextFactory  factory  boolean  JXPathContext  context  parentContext  HttpSession  session  HttpSessionAndServletContext  handle  
[BugLab_Variable_Misuse]^session.setAttribute ( Constants.JXPATH_CONTEXT, parentContext ) ;^170^^^^^157^173^session.setAttribute ( Constants.JXPATH_CONTEXT, context ) ;^[CLASS] JXPathServletContexts  [METHOD] getSessionContext [RETURN_TYPE] JXPathContext   HttpSession session ServletContext servletContext [VARIABLES] ServletContext  servletContext  JXPathContextFactory  factory  boolean  JXPathContext  context  parentContext  HttpSession  session  HttpSessionAndServletContext  handle  
[BugLab_Wrong_Operator]^if  ( context != null )  {^185^^^^^179^194^if  ( context == null )  {^[CLASS] JXPathServletContexts  [METHOD] getApplicationContext [RETURN_TYPE] JXPathContext   ServletContext servletContext [VARIABLES] ServletContext  servletContext  JXPathContextFactory  factory  boolean  JXPathContext  context  
[BugLab_Argument_Swapping]^context = servletContext.newContext ( null, factory ) ;^186^^^^^179^194^context = factory.newContext ( null, servletContext ) ;^[CLASS] JXPathServletContexts  [METHOD] getApplicationContext [RETURN_TYPE] JXPathContext   ServletContext servletContext [VARIABLES] ServletContext  servletContext  JXPathContextFactory  factory  boolean  JXPathContext  context  
[BugLab_Variable_Misuse]^new KeywordVariables ( null, servletContext )  ) ;^188^189^190^^^179^194^new KeywordVariables ( Constants.APPLICATION_SCOPE, servletContext )  ) ;^[CLASS] JXPathServletContexts  [METHOD] getApplicationContext [RETURN_TYPE] JXPathContext   ServletContext servletContext [VARIABLES] ServletContext  servletContext  JXPathContextFactory  factory  boolean  JXPathContext  context  
[BugLab_Argument_Swapping]^new KeywordVariables ( servletContext, Constants.APPLICATION_SCOPE )  ) ;^188^189^190^^^179^194^new KeywordVariables ( Constants.APPLICATION_SCOPE, servletContext )  ) ;^[CLASS] JXPathServletContexts  [METHOD] getApplicationContext [RETURN_TYPE] JXPathContext   ServletContext servletContext [VARIABLES] ServletContext  servletContext  JXPathContextFactory  factory  boolean  JXPathContext  context  
[BugLab_Argument_Swapping]^context.setVariables ( new KeywordVariables ( servletContext, Constants.APPLICATION_SCOPE )  ) ;^187^188^189^190^^179^194^context.setVariables ( new KeywordVariables ( Constants.APPLICATION_SCOPE, servletContext )  ) ;^[CLASS] JXPathServletContexts  [METHOD] getApplicationContext [RETURN_TYPE] JXPathContext   ServletContext servletContext [VARIABLES] ServletContext  servletContext  JXPathContextFactory  factory  boolean  JXPathContext  context  
